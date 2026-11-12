"""Uncertainty-aware P1 search combining triplet guidance and dual space.

The data term is a signed Gaussian *intensity* likelihood.  Approximate
direct-method relations are annealed search guidance.  The real-space step
alternates charge flipping with a finite atomic-support projection.  This is
an explicitly defined independent algorithm; it is not an implementation of
SHELXT's integrated space-group/element assignment procedure.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .direct_methods import (build_triplet_graph, intensity_amplitude_prox,
                             tangent_update, triplet_residual)
from .maps import DensityMap, grid_shape
from .scattering import atomic_form_factor, structure_factors


def merge_friedel_intensities(reflections, scale=1.0):
    """Merge independent measurements before constructing any amplitudes.

    Values and standard uncertainties are returned for consecutive ``h,-h``
    pairs.  Each pair is one independent observation in reported scores.
    Inverse-variance weights are rescaled within each bucket, avoiding
    ``1/sigma**2`` overflow.  All holdout labels are checked before exclusion.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Intensity scale must be finite and positive")
    buckets, labels = {}, {}
    for i, (h, y, sigma) in enumerate(zip(reflections.hkl, reflections.intensity, reflections.sigma)):
        key = min(tuple(h), tuple(-h))
        flag = bool(reflections.free_flags[i]) if reflections.free_flags is not None else False
        if key in labels and labels[key] != flag:
            raise ValueError("Holdout leakage: Friedel-related observations have different flags")
        labels[key] = flag
        if flag or not np.any(h):
            continue
        if key not in buckets:
            buckets[key] = [float(y), float(sigma), 1., 1]
            continue
        mean, minimum_sigma, weight_sum, count = buckets[key]
        if sigma < minimum_sigma:
            weight_sum *= (sigma / minimum_sigma)**2
            minimum_sigma = float(sigma)
        weight = (minimum_sigma / sigma)**2
        total = weight_sum + weight
        mean = mean * (weight_sum / total) + float(y) * (weight / total)
        buckets[key] = [mean, minimum_sigma, total, count + 1]
    if len(buckets) < 10:
        raise ValueError("P1 phase retrieval requires at least ten independent Friedel pairs")
    hkl, observed, sigma, multiplicity = [], [], [], []
    for h, (mean, minimum_sigma, weight_sum, count) in sorted(buckets.items()):
        hkl.extend((h, tuple(-v for v in h)))
        observed.extend((mean / scale,) * 2)
        sigma.extend((minimum_sigma / np.sqrt(weight_sum) / scale,) * 2)
        multiplicity.extend((count,) * 2)
    return (np.asarray(hkl, dtype=np.int64), np.asarray(observed),
            np.asarray(sigma), np.asarray(multiplicity, dtype=int))


def resolution_band(cell, shape, maximum_stol_sq):
    """Metric sphere on an FFT grid without an N-voxel by 3 index array."""
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(float) for n in shape]
    x, y, z = axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]
    metric = cell.reciprocal.T @ cell.reciprocal
    q2 = np.empty(shape, dtype=float)
    q2[:] = metric[0, 0] * x**2
    q2 += metric[1, 1] * y**2
    q2 += metric[2, 2] * z**2
    q2 += 2 * metric[0, 1] * x * y
    q2 += 2 * metric[0, 2] * x * z
    q2 += 2 * metric[1, 2] * y * z
    band = q2 <= 4 * maximum_stol_sq * (1 + 1e-9)
    # Even-grid Nyquist indices are their own modular negatives.  For a skew
    # metric a spherical band may otherwise retain one member of a conjugate
    # pair on these planes.  No measured index lies there (grid_shape enforces
    # strict oversampling), so exclude them and preserve a real density.
    for axis, size in enumerate(shape):
        if size % 2 == 0:
            plane = [slice(None)] * 3
            plane[axis] = size // 2
            band[tuple(plane)] = False
    return band


def _atomic_support(values, cell, count, minimum_distance):
    """Positive density inside 3x3x3 supports about metric-separated peaks."""
    peaks = DensityMap(values, cell).peaks(count, minimum_distance)
    mask = np.zeros(values.shape, dtype=bool)
    shape = np.asarray(values.shape)
    offsets = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), axis=-1).reshape(-1, 3)
    for peak in peaks:
        neighbours = (np.asarray(peak["grid_index"]) + offsets) % shape
        mask[tuple(neighbours.T)] = True
    return np.where(mask, np.maximum(values, 0), 0)


def _patterson_seeds(cell, shape, idx, observed, count, minimum_distance):
    reciprocal = np.zeros(shape, dtype=complex)
    reciprocal[idx] = observed
    patterson = np.fft.fftn(reciprocal).real
    # Signed measured intensities remain in the transform.  Positivity here
    # is only a real-space initialization constraint, not data modification.
    inverse_grid = np.ix_(*[(-np.arange(n)) % n for n in shape])
    patterson = (patterson + patterson[inverse_grid]) / 2
    peaks = DensityMap(patterson, cell, "normalized_patterson").peaks(2 * count + 1, minimum_distance)
    vectors = {}
    for peak in peaks:
        displacement = cell.minimum_image(peak["frac"])
        if np.linalg.norm(cell.to_cartesian(displacement)) >= minimum_distance:
            index = tuple(peak["grid_index"])
            opposite = tuple((-np.asarray(index)) % np.asarray(shape))
            key = min(index, opposite)
            vectors[key] = peak["value"]
    # A Patterson peak and its inverse are the same vector hypothesis.  The
    # canonical representative prevents rounding-order flips from changing
    # the initial model when the user changes intensity units.
    maximum = max(float(np.max(patterson)), 1e-30)
    ordered = sorted(vectors, key=lambda key: (-round(vectors[key] / maximum, 13), key))[:count]
    return np.maximum(patterson / maximum, 0), ordered


def solve_hybrid(cell, reflections, elements, options, scale):
    # Lazy import avoids changing the public PhasingResult contract.
    from .phasing import PhasingResult, _subgrid_peaks

    hkl, observed, uncertainty, multiplicity = merge_friedel_intensities(reflections, scale)
    shape = grid_shape(hkl, oversampling=options.oversampling, maximum_voxels=2_097_152)
    idx = tuple((hkl % shape).T)
    stol_sq = cell.stol_sq(hkl)
    # Sum by chemical species, not by atom: workspace O(N_reflection).
    envelope2 = np.zeros(len(hkl))
    for element, count in Counter(elements).items():
        envelope2 += count * atomic_form_factor(element, stol_sq)**2
    physical_envelope = np.sqrt(envelope2)
    envelope = np.maximum(physical_envelope, 0.08 * physical_envelope.max())
    if not options.sharpen:
        envelope[:] = physical_envelope.max()
    normalized_i = observed / envelope**2
    normalized_s = uncertainty / envelope**2
    target = np.sqrt(np.maximum(normalized_i, 0))
    # HKL intensity units are arbitrary.  Scale the algorithmic proximity
    # variance with the work amplitudes, so I,sigma -> c*(I,sigma) implies
    # every Fourier iterate -> sqrt(c)*F and leaves all inferred phases
    # unchanged.  Positive values here set units only; signed I stays in the
    # objective.  The fallback permits an all-negative diagnostic input.
    amplitude_scale2 = float(np.mean(np.maximum(normalized_i[::2], 0)))
    if amplitude_scale2 == 0:
        amplitude_scale2 = float(np.mean(normalized_s[::2]))
    proximity_variance = options.data_relaxation * amplitude_scale2
    reliability = np.maximum(normalized_i, 0) / np.hypot(normalized_i, normalized_s)
    graph = build_triplet_graph(hkl, target / np.sqrt(amplitude_scale2), reliability,
                                maximum_reflections=options.triplet_reflections)
    band = resolution_band(cell, shape, float(stol_sq.max()))
    rng = np.random.default_rng(options.seed)
    patterson, vectors = (None, [])
    if options.initialization != "random":
        patterson, vectors = _patterson_seeds(cell, shape, idx, normalized_i / amplitude_scale2,
                                            min(64, max(4, (options.trials + 1) // 2)),
                                            options.minimum_peak_distance)
    candidates, history = [], []
    denom = max(float(target[::2].sum()), 1e-14)
    # The guide vanishes after the first quarter of the search.  This is
    # deliberately a heuristic schedule, not a global-convergence claim.
    guide_iterations = max(1, min(100, options.iterations // 4))
    for trial in range(options.trials):
        use_patterson = bool(vectors) and (options.initialization == "patterson" or
                                          (options.initialization == "mixed" and trial % 2 == 1))
        if use_patterson:
            vector = vectors[(trial // 2) % len(vectors)]
            seed = np.minimum(patterson, np.roll(patterson, vector, axis=(0, 1, 2)))
            # Minimum superposition is centrosymmetric about half the vector.
            # An explicit perturbation releases that accidental initialization
            # symmetry; no atomic coordinates are used to break it.
            seed = seed + 0.02 * max(seed.std(), 1e-14) * rng.normal(size=shape)
            initialization = "patterson_minimum_superposition"
        else:
            seed = rng.normal(size=shape)
            initialization = "random_real_density"
        seed_spectrum = np.fft.ifftn(seed)
        phase = np.exp(1j * np.angle(seed_spectrum[idx]))
        # Initial missing amplitudes are unknown, not inherited in arbitrary
        # units from a random or Patterson density. They evolve after step 1.
        f = np.zeros(shape, dtype=complex)
        f[idx] = target * phase
        del seed, seed_spectrum
        f[~band] = 0
        f[0, 0, 0] = 0
        best_score, best_f = np.inf, f.copy()
        best_iteration = 0
        for iteration in range(options.iterations):
            rho = np.fft.fftn(f).real
            atomic_step = (options.atomic_interval > 0 and
                           (iteration + 1) % options.atomic_interval == 0)
            if atomic_step:
                modified = _atomic_support(rho, cell, len(elements), options.minimum_peak_distance)
            else:
                delta = options.threshold_sigma * rho.std()
                modified = np.where(rho < delta, -rho, rho)
            g = np.fft.ifftn(modified)
            magnitude = np.abs(g[idx])
            # The checkpoint is judged before the data step.  Otherwise hard
            # projection would give a zero residual for every random phase set.
            sigma_rmse = float(np.sqrt(np.mean(((magnitude[::2]**2 - normalized_i[::2]) /
                                                normalized_s[::2])**2)))
            score = float(np.sum(np.abs(magnitude[::2] - target[::2])) / denom)
            phase = np.angle(g[idx])
            # Fix roundoff phase indeterminacy of exactly extinct coefficients.
            phase[1::2] = -phase[::2]
            if iteration < guide_iterations and iteration % 5 == 0 and options.triplet_strength:
                phase = tangent_update(phase, graph,
                                       options.triplet_strength * (1 - iteration / guide_iterations))
            radial = intensity_amplitude_prox(magnitude, normalized_i, normalized_s,
                                              proximity_variance)
            g[idx] = radial * np.exp(1j * phase)
            g[~band] = 0
            f = g
            if sigma_rmse < best_score:
                best_score, best_f = sigma_rmse, f.copy()
                best_iteration = iteration + 1
            if iteration % 25 == 0 or iteration == options.iterations - 1:
                history.append({"trial": trial, "iteration": iteration + 1,
                                "amplitude_projection_residual": score,
                                "intensity_sigma_rmse": sigma_rmse,
                                "best_intensity_sigma_rmse": best_score,
                                "triplet_residual": triplet_residual(phase, graph),
                                "atomic_support_step": atomic_step})
        # The atomic map uses the physical fall-off for observed coefficients.
        # Missing grid coefficients are model latent variables.  Zero them in
        # this diagnostic extraction map instead of claiming measured evidence.
        display = np.zeros(shape, dtype=complex)
        display[idx] = best_f[idx] * envelope
        density = DensityMap(np.fft.fftn(display).real / cell.volume, cell, "hybrid_phase_retrieval")
        model = _subgrid_peaks(density, elements, options.minimum_peak_distance)
        if model is not None:
            predicted = np.abs(structure_factors(model, hkl[::2]))**2
            pw, yw = predicted / uncertainty[::2], observed[::2] / uncertainty[::2]
            denominator = float(pw @ pw)
            atomic_scale = max(0., float(pw @ yw / denominator)) if denominator > 0 else 0.
            atomic_rmse = float(np.sqrt(np.mean((atomic_scale * pw - yw)**2)))
            candidates.append({"trial": trial, "projection_residual": best_score,
                               "projection_residual_kind": "pre-proximal intensity sigma RMSE",
                               "atomic_sigma_rmse": atomic_rmse,
                               "profiled_scale": atomic_scale * scale,
                               "best_iteration": best_iteration, "structure": model,
                               "grid_shape": list(shape), "algorithm": "hybrid",
                               "initialization": initialization,
                               "triplet_graph": graph.to_dict()})
    candidates.sort(key=lambda candidate: candidate["atomic_sigma_rmse"])
    result = PhasingResult(candidates[:options.keep], history, options.seed)
    result.assumptions += [
        "Signed independent Gaussian intensity errors; detector-count data must first be reduced appropriately",
        "Triplet relations guide early search; their weight vanishes before final iterations",
        "Missing coefficients are latent search variables and are never exported as observations",
        "Candidate atomic_sigma_rmse uses work observations and is not an independent validation score",
    ]
    result.diagnostics = {
        "algorithm": "hybrid", "independent_work_pairs": len(hkl) // 2,
        "negative_merged_work_intensities": int(np.count_nonzero(observed[::2] < 0)),
        "work_observation_count": int(multiplicity[::2].sum()),
        "grid_shape": list(shape), "triplet_graph": graph.to_dict(),
        "patterson_seed_vectors": len(vectors),
        "reciprocal_update": "global signed-intensity radial proximal minimizer",
        "data_relaxation": options.data_relaxation,
        "effective_proximity_variance": proximity_variance,
        "intensity_unit_equivariance": "I,sigma multiplied by c gives Fourier iterates multiplied by sqrt(c)",
        "normalization": "composition scattering envelope with 8 percent floor",
        "selection": "work-only profiled atomic intensity residual; independent validation is external",
    }
    return result
