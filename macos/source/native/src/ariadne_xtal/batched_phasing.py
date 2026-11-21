"""Batched charge-flipping proposals with optional device-resident FFTs."""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from time import perf_counter

import numpy as np

from .hybrid_phasing import merge_friedel_intensities, resolution_band
from .maps import DensityMap, grid_shape
from .phasing import _subgrid_peaks
from .scattering import atomic_form_factor


def _project(xp, spectrum, measured, lookup, band, target):
    values = spectrum[:, measured]
    magnitude = xp.abs(values)
    phase = values / xp.where(magnitude > 1e-15, magnitude, 1.)
    observed = target * xp.where(magnitude > 1e-15, phase, 1.)
    return xp.where(band, xp.where(lookup >= 0, observed[:, xp.maximum(lookup, 0)], spectrum), 0.)


def _iterate(xp, initial, measured, lookup, band, target, options, iterations, synchronize):
    axes = (-3, -2, -1)
    shape = initial.shape
    f = _project(xp, initial.reshape(shape[0], -1), measured, lookup, band, target)
    best = f
    best_score = xp.full((shape[0],), float('inf'), dtype=target.dtype)
    denom = xp.maximum(xp.sum(target), 1e-14)
    for _ in range(iterations):
        rho = xp.real(xp.fft.fftn(f.reshape(shape), axes=axes))
        delta = options.threshold_sigma * xp.std(rho, axis=axes, keepdims=True)
        g = xp.fft.ifftn(xp.where(rho < delta, -rho, rho), axes=axes).reshape(shape[0], -1)
        score = xp.sum(xp.abs(xp.abs(g[:, measured]) - target), axis=1) / denom
        f = _project(xp, g, measured, lookup, band, target)
        better = score < best_score
        best = xp.where(better[:, None], f, best)
        best_score = xp.where(better, score, best_score)
        if synchronize is not None:
            synchronize(f, best, best_score)
    return best[:, measured], best_score


def batched_proposals(cell, reflections, elements, options, seeds, iterations, *,
                      scale=1., backend='batched_cpu', batch_size=16):
    """Return one atom proposal per seed using shared work amplitudes and bands."""
    if backend not in {'batched_cpu', 'metal', 'cuda'} or type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError('Specify batched_cpu, metal or cuda and a batch size in 1..256')
    if backend == 'metal':
        try:
            import mlx.core as xp
        except ImportError as exc:
            raise RuntimeError('Install MLX in the Apple Silicon Python environment to enable Metal phasing') from exc
        if not xp.metal.is_available():
            raise RuntimeError('Metal is unavailable in this process')
        context = xp.stream(xp.gpu)
        to_device, to_cpu = xp.array, np.asarray
        synchronize, sync_iteration = xp.eval, xp.eval
    elif backend == 'cuda':
        try:
            import cupy as xp
        except ImportError as exc:
            raise RuntimeError('Install CuPy and a compatible CUDA runtime to use NVIDIA phasing') from exc
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA is unavailable in this process')
        context = xp.cuda.Device()
        to_device, to_cpu = xp.asarray, xp.asnumpy

        def synchronize(*arrays):
            xp.cuda.get_current_stream().synchronize()

        sync_iteration = None
    else:
        xp, context = np, nullcontext()
        to_device, to_cpu = np.asarray, np.asarray
        synchronize = sync_iteration = None
    real_type, complex_type = (np.float64, np.complex128) if xp is np else (np.float32, np.complex64)

    hkl, observed, _, _ = merge_friedel_intensities(reflections, scale)
    shape = grid_shape(hkl, oversampling=options.oversampling, maximum_voxels=2_097_152)
    voxels = int(np.prod(shape))
    measured = np.ravel_multi_index((hkl % shape).T, shape).astype(np.int32)
    lookup = np.full(voxels, -1, dtype=np.int32)
    lookup[measured] = np.arange(len(hkl), dtype=np.int32)
    s2 = cell.stol_sq(hkl)
    power = np.zeros(len(hkl))
    for element, count in Counter(elements).items():
        power += count * atomic_form_factor(element, s2)**2
    envelope = np.sqrt(power)
    sharpen = np.maximum(envelope / envelope.max(), .08) if options.sharpen else np.ones_like(envelope)
    amplitudes = np.sqrt(np.maximum(observed, 0.)) / sharpen
    band = resolution_band(cell, shape, float(s2.max())).ravel()
    band[0] = True

    # Bound simultaneous grids independently of the requested trial cohort.
    actual_batch = min(len(seeds), batch_size, max(1, 2_097_152 // voxels))
    models = []
    upload_bytes = download_bytes = 0
    transfer_seconds = 0.
    with context:
        started = perf_counter()
        selected, locations, allowed = map(to_device, (measured, lookup, band))
        target = to_device(amplitudes.astype(real_type))
        if synchronize is not None:
            synchronize(selected, locations, allowed, target)
            transfer_seconds += perf_counter() - started
            upload_bytes += measured.nbytes + lookup.nbytes + band.nbytes + target.nbytes
        for start in range(0, len(seeds), actual_batch):
            group = seeds[start:start+actual_batch]
            initial = np.stack([np.fft.ifftn(np.random.default_rng(seed).normal(size=shape))
                                for seed in group]).astype(complex_type)
            initial.reshape(len(group), -1)[:, 0] = 0.
            started = perf_counter()
            device_initial = to_device(initial)
            if synchronize is not None:
                synchronize(device_initial)
                transfer_seconds += perf_counter() - started
                upload_bytes += initial.nbytes
            coefficients, residuals = _iterate(xp, device_initial, selected, locations,
                                               allowed, target, options, iterations, sync_iteration)
            if synchronize is not None:
                synchronize(coefficients, residuals)
            started = perf_counter()
            best, scores = to_cpu(coefficients), to_cpu(residuals)
            if synchronize is not None:
                transfer_seconds += perf_counter() - started
                download_bytes += best.nbytes + scores.nbytes
            for local, (coefficients, score) in enumerate(zip(best, scores)):
                display = np.zeros(voxels, dtype=complex)
                display[measured] = coefficients * sharpen
                density = DensityMap(np.fft.fftn(display.reshape(shape)).real / cell.volume,
                                     cell, 'coarse_charge_flip')
                models.append(dict(trial=start+local, projection_residual=float(score),
                                   structure=_subgrid_peaks(density, elements, options.minimum_peak_distance)))
    return models, {'backend': backend, 'algorithm': 'batched_charge_flip',
                    'precision': np.dtype(real_type).name, 'grid_shape': list(shape),
                    'batch_size': actual_batch, 'upload_bytes': upload_bytes,
                    'download_bytes': download_bytes, 'transfer_seconds': transfer_seconds}
