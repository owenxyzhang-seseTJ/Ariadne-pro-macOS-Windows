"""Direct dual-space proposals for an exactly overlapping, two-domain twin."""
from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np

from .batched_phasing import fft_backend
from .hybrid_phasing import _patterson_seeds, merge_friedel_intensities, resolution_band
from .maps import DensityMap, grid_shape
from .phasing import PhasingOptions, _subgrid_peaks
from .scattering import atomic_form_factor, structure_factors
from .twin_diagnostics import IDENTITY, _matrix, exact_metric, twin_screen


def _orbit_problem(cell, reflections, law, elements, options):
    hkl, y, sigma, _ = merge_friedel_intensities(reflections)
    unique = {tuple(h): (v, s) for h, v, s in zip(hkl[::2], y[::2], sigma[::2])}
    paired = set()
    orbits = []
    for h in sorted(unique):
        if h in paired:
            continue
        transformed = tuple(law @ h)
        mate = min(transformed, tuple(-v for v in transformed))
        orbit = tuple(sorted((h, mate)))
        orbits.append(orbit)
        paired.update(orbit)
    orbit_hkl = np.array(orbits, dtype=np.int64)
    singles = np.all(orbit_hkl[:, 0] == orbit_hkl[:, 1], axis=1)
    observed = np.zeros((len(orbits), 2))
    sd = np.full_like(observed, np.inf)
    for i, (left, right) in enumerate(orbits):
        for j, key in enumerate((left, right)):
            if (j == 0 or left != right) and key in unique:
                observed[i, j], sd[i, j] = unique[key]
    positive = observed[observed > 0]
    unit = float(np.median(positive)) if len(positive) else float(np.median(sd[np.isfinite(sd)]))
    s2 = cell.stol_sq(orbit_hkl[:, 0])
    power = np.zeros(len(orbits))
    for element, count in Counter(elements).items():
        power += count * atomic_form_factor(element, s2)**2
    envelope = np.maximum(np.sqrt(power/power.max()), .08) if options.sharpen else np.ones(len(s2))
    normalization = unit * envelope[:, None]**2
    response = observed / normalization
    # The duplicate Friedel sign is only a Fourier conjugacy constraint.
    # A missing twin mate carries exactly zero statistical weight.
    relative_sd = sd/normalization
    weight = np.zeros_like(relative_sd)
    observed_mask = np.isfinite(relative_sd)
    weight[observed_mask] = 1/relative_sd[observed_mask]**2
    if not np.isfinite(weight).all():
        raise ValueError("Standard uncertainties span an unsupported numerical range")
    flat = orbit_hkl.reshape(-1, 3)
    shape = grid_shape(flat, oversampling=options.oversampling, maximum_voxels=2_097_152)
    voxels = int(np.prod(shape))
    measured = np.ravel_multi_index((flat % shape).T, shape).astype(np.int32)
    conjugate = np.ravel_multi_index((-flat % shape).T, shape).astype(np.int32)
    lookup = np.full(voxels, -1, dtype=np.int32)
    active_coefficients = np.flatnonzero(~np.column_stack((np.zeros(len(singles), bool), singles)).ravel())
    lookup[measured[active_coefficients]] = active_coefficients
    lookup[conjugate[active_coefficients]] = active_coefficients
    negative = np.zeros(voxels, dtype=bool)
    negative[conjugate[active_coefficients]] = True
    band = resolution_band(cell, shape, float(np.max(s2))).ravel()
    return shape, measured, lookup, negative, band, singles, response, weight, envelope, observed_mask


def orbit_proximal(xp, power, target, weight, fraction, singleton, strength):
    """Nonnegative quadratic intensity proximal step, batched over 2-site orbits.

    This solves exactly the four active sets of a convex problem in intensities.
    Phases are supplied by the real-space iterate and do not enter this step.
    """
    a = xp.where(singleton, 0., fraction)
    b = 1.-a
    y0, y1 = target[..., 0], target[..., 1]
    w0, w1 = weight[..., 0], weight[..., 1]
    g0 = strength + w0*b*b + w1*a*a
    g1 = strength + w0*a*a + w1*b*b
    cross = (w0+w1)*a*b
    rhs0 = strength*power[..., 0] + w0*b*y0 + w1*a*y1
    rhs1 = strength*power[..., 1] + w0*a*y0 + w1*b*y1
    det = strength*(strength + (w0+w1)*(a*a+b*b)) + w0*w1*(1.-2*a)**2
    x0 = (g1*rhs0 - cross*rhs1)/det
    x1 = (g0*rhs1 - cross*rhs0)/det
    feasible = (x0 >= 0) & (x1 >= 0)
    left, right = xp.maximum(rhs0, 0.), xp.maximum(rhs1, 0.)
    choose_left = left**2/g0 >= right**2/g1
    return xp.stack((xp.where(feasible, x0, xp.where(choose_left, left/g0, 0.)),
                     xp.where(feasible, x1, xp.where(choose_left, 0., right/g1))), axis=-1)


def _score_grid(xp, power, target, weight, fraction, singleton, estimate_fraction):
    a = xp.where(singleton, 0., fraction)
    if estimate_fraction:
        contrast = power[..., ::-1] - power
        contrast = xp.where(singleton[..., None], 0., contrast)
        normalizer = xp.sum(weight * contrast**2, axis=(-2, -1), keepdims=False)
        direction = xp.sum(weight * contrast*(target-power), axis=(-2, -1))
        fitted = xp.clip(direction/xp.maximum(normalizer, 1e-14), 0., .5)
        a = xp.where(singleton, 0., xp.where(normalizer > 1e-14, fitted, fraction[:, 0])[:, None])
    sum_model = (1.-a[..., None])*power + a[..., None]*power[..., ::-1]
    score = xp.sum(weight*(sum_model-target)**2, axis=(-2, -1))
    # A singleton may be first; retain the actual per-trial fraction separately.
    fitted = xp.max(a, axis=-1, keepdims=True)
    return score, fitted


def _fft_orbits(xp, initial, measured, lookup, negative, band, singleton,
                target, weight, options, iterations, alpha, fit_alpha, sync):
    shape = initial.shape
    size = shape[0]
    axes = (-3, -2, -1)
    f = initial.reshape(size, -1)
    fraction = alpha.reshape((size, 1))

    def update(values, spectrum, fraction):
        magnitude = xp.abs(values)
        unit_phase = xp.where(magnitude > 1e-15, values/xp.maximum(magnitude, 1e-15), 1.)
        fitted = orbit_proximal(xp, magnitude.reshape(size, -1, 2)**2, target, weight, fraction,
                                singleton, 1./options.data_relaxation)
        coefficients = xp.sqrt(fitted).reshape(size, -1) * unit_phase
        mapped = coefficients[:, xp.maximum(lookup, 0)]
        mapped = xp.where(negative, xp.conj(mapped), mapped)
        return xp.where(band, xp.where(lookup >= 0, mapped, spectrum), 0.)

    f = update(f[:, measured], f, fraction)
    best = f[:, measured]
    best_score = xp.full((size,), float('inf'), dtype=target.dtype)
    best_fraction = fraction[:, 0]
    for step in range(iterations):
        density = xp.real(xp.fft.fftn(f.reshape(shape), axes=axes))
        threshold = options.threshold_sigma * xp.std(density, axis=axes, keepdims=True)
        g = xp.fft.ifftn(xp.where(density < threshold, -density, density), axes=axes).reshape(size, -1)
        values = g[:, measured]
        score, fraction = _score_grid(xp, xp.abs(values.reshape(size, -1, 2))**2, target,
                                      weight, fraction, singleton, fit_alpha and step >= 4)
        f = update(values, g, fraction)
        better = score < best_score
        best = xp.where(better[:, None], f[:, measured], best)
        best_score = xp.where(better, score, best_score)
        best_fraction = xp.where(better, fraction[:, 0], best_fraction)
        if sync is not None:
            sync(f, fraction, best, best_score, best_fraction)
    return best, best_score, best_fraction


def direct_twin_phasing(cell, reflections, elements, law, options=None, *, assume_friedel=False,
                        fraction=.35, refine_fraction=True, backend='batched_cpu',
                        coarse_iterations=45, promote=3, inspect_rejections=1, batch_size=16):
    """Propose structures from signed, undetwinned, exact two-domain mixtures.

    Related observed indices must already be kept together by the caller.
    GPU proposals are screened and replayed on the double-precision CPU.
    """
    started = perf_counter()
    law = _matrix(law)
    if not assume_friedel:
        raise ValueError("Direct twin phasing requires declared Friedel equivalence")
    if not exact_metric(cell, law) or not any(np.array_equal(law @ law, sign*IDENTITY) for sign in (1, -1)):
        raise ValueError("Direct phase search supports an exact involution modulo Friedel equivalence")
    if np.array_equal(law, IDENTITY) or np.array_equal(law, -IDENTITY):
        raise ValueError("The law has no measurable contrast under Friedel equivalence")
    if not isinstance(elements, list) or not 1 <= len(elements) <= 512:
        raise ValueError("Supply the complete unit-cell element list")
    options = PhasingOptions(trials=8, iterations=220) if options is None else (
        options if isinstance(options, PhasingOptions) else PhasingOptions(**options))
    if (not 0 <= fraction <= .5 or type(refine_fraction) is not bool or
            backend not in {'batched_cpu', 'metal', 'cuda'} or
            any(type(v) is not int or v < lower for v, lower in
                ((coarse_iterations, 1), (promote, 1), (inspect_rejections, 0), (batch_size, 1))) or
            batch_size > 256 or options.trials * options.iterations > 2_000_000):
        raise ValueError("Invalid twin phase-search budget, backend or fraction")
    if reflections.free_flags is not None:
        roles = {}
        for h, mapped, flag in zip(reflections.hkl, reflections.hkl @ law.T, reflections.free_flags):
            key = min(tuple(h), tuple(-h), tuple(mapped), tuple(-mapped))
            if key in roles and roles[key] != flag:
                raise ValueError("Holdout flags split a two-domain reflection orbit")
            roles[key] = flag
    observations = reflections if reflections.free_flags is None else reflections.subset(~reflections.free_flags)
    geometry = _orbit_problem(cell, observations, law, elements, options)
    shape, measured, lookup, negative, band, singleton, target, weight, envelope, present = geometry
    if np.all(singleton):
        raise ValueError("No measured orbit distinguishes the proposed twin law")
    seeds = [int(v) for v in np.random.SeedSequence(options.seed).generate_state(options.trials)]
    start_fractions = np.full(options.trials, fraction)
    both = np.all(present, axis=1)
    if refine_fraction and np.sum(both) >= 20:
        variance = np.sum(1./weight[both], axis=1)
        normalized_difference = np.diff(target[both], axis=1).ravel()**2 / variance
        if float(np.mean(normalized_difference)) < 1.5:
            start_fractions[::2] = .48

    mixture_patterson = None
    vectors = []
    if options.initialization != 'random':
        selected = lookup >= 0
        measured_targets = np.where(present, target, 0.).ravel()
        idx = np.unravel_index(np.flatnonzero(selected), shape)
        mixture_patterson, vectors = _patterson_seeds(cell, shape, idx, measured_targets[lookup[selected]],
                                                     min(32, options.trials), options.minimum_peak_distance)

    def initial_density(trial):
        rng = np.random.default_rng(seeds[trial])
        perturbation = rng.normal(size=shape)
        if vectors and (options.initialization == 'patterson' or trial % 2 == 1):
            shift = vectors[trial//2 % len(vectors)]
            seed = np.minimum(mixture_patterson, np.roll(mixture_patterson, shift, axis=(0, 1, 2)))
            deviation = max(float(np.std(seed)), 1e-14)
            return (seed-np.mean(seed))/deviation + .035*perturbation
        return perturbation

    def run(indices, device, iterations):
        xp, context, to_device, to_cpu, synchronize, sync_iteration = fft_backend(device)
        real = np.float64 if xp is np else np.float32
        complex_type = np.complex128 if xp is np else np.complex64
        voxels = int(np.prod(shape))
        batch = min(batch_size, max(1, 2_097_152 // voxels))
        results = []
        with context:
            args = list(map(to_device, (measured, lookup, negative, band, singleton,
                                       target.astype(real), weight.astype(real))))
            scale = np.sqrt(np.maximum(np.sum(np.maximum(target, 0.), axis=-1) /
                                       np.maximum(np.sum(present, axis=-1), 1), 1e-8))
            normalization = float(np.sqrt(np.sum(scale**2)/voxels))
            for start in range(0, len(indices), batch):
                subset = indices[start:start+batch]
                initial = np.stack([np.fft.ifftn(initial_density(i)) for i in subset]).astype(complex_type)
                initial *= normalization * np.sqrt(voxels)
                best, score, alpha = _fft_orbits(xp, to_device(initial), *args, options,
                                                iterations, to_device(start_fractions[subset].astype(real)),
                                                refine_fraction, sync_iteration)
                if synchronize is not None:
                    synchronize(best, score, alpha)
                for i, values, loss, rate in zip(subset, to_cpu(best), to_cpu(score), to_cpu(alpha)):
                    results.append((int(i), values, float(loss/present.sum()), float(rate)))
        return results

    trials = list(range(options.trials))
    rejected = []
    inspection = []
    coarse = []
    coarse_seconds = 0.
    if backend != 'batched_cpu':
        coarse_start = perf_counter()
        proposal = run(trials, backend, min(options.iterations, coarse_iterations))
        coarse_seconds = perf_counter()-coarse_start
        coarse = [{'trial': i, 'score': score} for i, _, score, _ in sorted(proposal, key=lambda row: (row[2], row[0]))]
        trials = [v['trial'] for v in coarse[:promote]]
        rejected = [v['trial'] for v in coarse[promote:]]
        rng = np.random.default_rng(options.seed+1)
        inspection = rng.choice(rejected, size=min(len(rejected), inspect_rejections), replace=False).tolist()
    active = lookup >= 0
    coefficient_slots = lookup[active]
    widths = np.repeat(envelope, 2)
    def reconstruct(indices):
        candidates = []
        for i, values, loss, alpha in run(indices, 'batched_cpu', options.iterations):
            display = np.zeros(lookup.size, dtype=np.complex128)
            selected = (values*widths)[coefficient_slots]
            display[active] = np.where(negative[active], selected.conj(), selected)
            density = DensityMap(np.fft.fftn(display.reshape(shape)).real/cell.volume, cell, 'twin_mixture_phasing')
            model = _subgrid_peaks(density, elements, options.minimum_peak_distance)
            if model is None:
                continue
            if refine_fraction:
                fitted = twin_screen(model, observations, [law.tolist()])['rows'][1]
            else:
                left = np.abs(structure_factors(model, observations.hkl))**2
                right = np.abs(structure_factors(model, observations.hkl @ law.T))**2
                weighted = ((1-fraction)*left + fraction*right)/observations.sigma
                y = observations.intensity/observations.sigma
                scale = max(0., float(weighted @ y/max(weighted @ weighted, 1e-30)))
                residual = weighted*scale-y
                fitted = {'scale': scale, 'fraction': fraction, 'chi_square': float(residual @ residual)}
            candidates.append(dict(trial=i, structure=model, likelihood=fitted, projection_loss=loss,
                                   grid_fraction=alpha, sigma_rmse=float(np.sqrt(fitted['chi_square']/len(observations)))))
        candidates.sort(key=lambda row: (row['sigma_rmse'], row['trial']))
        return candidates
    replay_start = perf_counter()
    candidates = reconstruct(trials)
    replay_seconds = perf_counter()-replay_start
    inspect_start = perf_counter()
    inspected = reconstruct(inspection) if inspection else []
    inspection_seconds = perf_counter()-inspect_start
    elapsed = perf_counter()-started
    return {'schema': 'ariadne-direct-twin-phasing/1', 'backend': backend, 'grid_shape': list(shape),
            'observed_orbit_rows': int(present.sum()), 'orbits': len(singleton),
            'singletons': int(singleton.sum()), 'missing_orbit_rows': int((~present & ~singleton[:, None]).sum()),
            'seeds': seeds, 'patterson_seed_vectors': len(vectors), 'initial_fractions': start_fractions.tolist(), 'coarse': coarse, 'rejected_trials': rejected, 'inspected_trials': inspection,
            'candidates': candidates[:options.keep], 'audited_rejections': inspected,
            'runtime': {'coarse_seconds': coarse_seconds, 'cpu_replay_seconds': replay_seconds,
                        'inspection_seconds': inspection_seconds, 'search_seconds': elapsed-inspection_seconds},
            'scope': 'Working intensities only. Chemical and independent predictive validation remain separate.'}
