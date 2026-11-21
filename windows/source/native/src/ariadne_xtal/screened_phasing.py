"""Coarse phase-search ranking with an independent double-precision recheck."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
import numpy as np

from .model import Reflections
from .phasing import PhasingOptions, solve_p1
from .scattering import calculated_intensities


def _score(candidate, reflections):
    observed = reflections.intensity / reflections.sigma
    prediction = calculated_intensities(candidate['structure'], reflections.hkl) / reflections.sigma
    norm = prediction @ prediction
    scale = max(0., float(observed @ prediction / norm)) if norm > 0 else 0.
    residual = observed - scale * prediction
    return float(np.mean(residual**2))


def screened_phase_search(cell, reflections, elements, options=None, *, coarse_iterations=40,
                          promoted=4, random_promotions=1, resolution_quantile=.65,
                          inspected_rejections=1, scale=1., coarse_backend='legacy_cpu',
                          batch_size=16):
    """Screen seeded starts at lower resolution, then rerun promoted starts on CPU."""
    options = options or PhasingOptions(algorithm='hybrid', trials=12)
    if (type(coarse_iterations) is not int or not 1 <= coarse_iterations <= options.iterations
            or type(promoted) is not int or not 1 <= promoted < options.trials
            or type(random_promotions) is not int or not 0 <= random_promotions < promoted
            or type(inspected_rejections) is not int or not 0 <= inspected_rejections <= options.trials - promoted
            or not .2 <= resolution_quantile < 1.
            or coarse_backend not in {'legacy_cpu', 'batched_cpu', 'metal', 'cuda'}):
        raise ValueError('Invalid screened search budgets or resolution quantile')
    use = (np.ones(len(reflections), dtype=bool) if reflections.free_flags is None else ~reflections.free_flags)
    if reflections.free_flags is not None:
        work_keys = {min(tuple(h), tuple(-h)) for h in reflections.hkl[use]}
        if any(min(tuple(h), tuple(-h)) in work_keys for h in reflections.hkl[~use]):
            raise ValueError('Holdout leakage: Friedel-related observations have different flags')
    work = Reflections(reflections.hkl[use], reflections.intensity[use], reflections.sigma[use])
    resolution = cell.stol_sq(work.hkl)
    cut = resolution <= np.quantile(resolution, resolution_quantile)
    coarse = Reflections(work.hkl[cut], work.intensity[cut], work.sigma[cut])
    rng = np.random.default_rng(options.seed)
    seeds = [int(s) for s in rng.integers(0, 2**31, size=options.trials)]
    ranking = []
    started = perf_counter()
    if coarse_backend == 'legacy_cpu':
        screening = {'backend': coarse_backend, 'algorithm': options.algorithm, 'precision': 'float64'}
        for trial, seed in enumerate(seeds):
            result = solve_p1(cell, coarse, elements,
                              replace(options, trials=1, keep=1, iterations=coarse_iterations, seed=seed), scale)
            score = _score(result.candidates[0], coarse) if result.candidates else np.inf
            ranking.append((score, trial))
    else:
        from .batched_phasing import batched_proposals

        proposals, screening = batched_proposals(cell, coarse, elements, options, seeds,
                                                 coarse_iterations, scale=scale,
                                                 backend=coarse_backend, batch_size=batch_size)
        ranking = [(_score(candidate, coarse) if candidate['structure'] is not None else np.inf,
                    candidate['trial']) for candidate in proposals]
    screening['wall_seconds'] = perf_counter() - started
    ranking.sort()
    accepted = [t for _, t in ranking[:promoted-random_promotions]]
    if random_promotions:
        others = [t for _, t in ranking if t not in accepted]
        accepted.extend(int(t) for t in rng.choice(others, size=random_promotions, replace=False))
    rejected = [t for _, t in ranking if t not in accepted][:inspected_rejections]
    candidates, rejected_models, diagnostics = [], [], []
    recheck_started = perf_counter()
    for trial in accepted + rejected:
        trial_started = perf_counter()
        result = solve_p1(cell, work, elements, replace(options, trials=1, keep=1, seed=seeds[trial]), scale)
        candidate = result.candidates[0] if result.candidates else None
        score = _score(candidate, work) if candidate else None
        diagnostics.append(dict(trial=trial, seed=seeds[trial], promoted=trial in accepted,
                                full_work_score=score, backend='cpu_float64',
                                wall_seconds=perf_counter()-trial_started))
        if candidate:
            collection = candidates if trial in accepted else rejected_models
            collection.append({**candidate, 'trial': trial, 'full_work_score': score})
    candidates.sort(key=lambda item: item['full_work_score'])
    return {'candidates': candidates[:options.keep], 'trial_diagnostics': diagnostics,
            'screening': [{'trial': t, 'coarse_work_score': float(s) if np.isfinite(s) else None}
                          for s, t in ranking],
            'screening_runtime': screening, 'full_recheck_wall_seconds': perf_counter()-recheck_started,
            'audited_rejections': rejected_models,
            'full_work_reflections': len(work), 'coarse_work_reflections': len(coarse),
            'iteration_budget': options.trials * coarse_iterations + (promoted+inspected_rejections)*options.iterations}
