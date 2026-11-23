"""Compare blind two-domain phase proposals at fixed work and selection budgets."""
from __future__ import annotations

import argparse
import copy
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ariadne_xtal.autonomy import evidence_partition
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.phasing import PhasingOptions, solve_p1
from ariadne_xtal.refinement import RefinementOptions, refine
from ariadne_xtal.scattering import calculated_intensities
from ariadne_xtal.twin_phasing import direct_twin_phasing

from benchmark_blind_phasing import matched_rms


LAW = np.diag([1, -1, -1])
RMS_CUTOFF = .25


def cases():
    for i, (length, alpha) in enumerate(((6.1, .0), (6.8, .20), (7.5, .45), (9.2, .49), (10.5, .50))):
        rng = np.random.default_rng(31017+i)
        cell = Cell(length, length+.63, length+.85)
        atoms = []
        for element in ('Fe', 'Cl', 'O', 'O'):
            p = rng.random(3)
            while any(np.linalg.norm(cell.to_cartesian(cell.minimum_image(p-a.frac))) < 1.1 for a in atoms):
                p = rng.random(3)
            atoms.append(Atom(f'{element}{len(atoms)+1}', element, p, u_iso=.018))
        model = Structure(cell, atoms)
        cutoff = .85
        hkl = np.array(list(product(*(range(-int(a/cutoff), 1+int(a/cutoff)) for a in (cell.a, cell.b, cell.c)))))
        hkl = hkl[np.any(hkl, axis=1)]
        hkl = hkl[cell.d_spacing(hkl) >= cutoff]
        intensity = calculated_intensities(model, hkl, twin_laws=[LAW], twin_fractions=[1-alpha, alpha])
        sigma = np.hypot(8., .035*intensity)
        y = intensity + sigma*rng.normal(size=len(sigma))
        use = rng.random(len(y)) > .07
        yield model, Reflections(hkl[use], y[use], sigma[use]), alpha


def score_candidates(candidates, work, selection, audit, reference, iterations):
    transformed = copy.deepcopy(reference)
    for atom in transformed.atoms:
        atom.frac = (LAW.T @ atom.frac) % 1
    fits = []
    for item in candidates:
        start = item.get('likelihood', {})
        fraction = float(np.clip(start.get('fraction') if start.get('fraction') is not None else .25, 1e-5, 1-1e-5))
        result = refine(item['structure'], work, RefinementOptions(max_nfev=iterations,
                        scale=max(start.get('scale', 1.), 1e-12), twin_laws=[LAW],
                        twin_fractions=[1-fraction, fraction], refine_twin_fractions=True))
        pred = calculated_intensities(result.structure, selection.hkl, scale=result.scale,
                                      twin_laws=[LAW], twin_fractions=result.twin_fractions)
        loss = float(np.mean(((pred-selection.intensity)/selection.sigma)**2))
        fits.append((loss, item['trial'], result))
    if not fits:
        return {'candidates': 0}
    fits.sort(key=lambda x: (x[0], x[1]))
    _, best_id, best = fits[0]
    audit_error = None
    if audit is not None:
        prediction = calculated_intensities(best.structure, audit.hkl, scale=best.scale,
                                            twin_laws=[LAW], twin_fractions=best.twin_fractions)
        audit_error = float(np.sqrt(np.mean(((prediction-audit.intensity)/audit.sigma)**2)))
    rmsd = [min(matched_rms(reference, fit.structure), matched_rms(transformed, fit.structure)) for _, _, fit in fits]
    return {'candidates': len(fits), 'selected_trial': best_id, 'selected_rmsd_angstrom': rmsd[0],
            'best_recovery_rmsd_angstrom': min(rmsd), 'recoverable_candidates': sum(x <= RMS_CUTOFF for x in rmsd),
            'selected_recovered': rmsd[0] <= RMS_CUTOFF,
            'selected_fraction_unlabelled': float(np.min(best.twin_fractions)),
            'audit_sigma_rmse': audit_error}


def benchmark(args):
    cases_out = []
    for i, (truth, r, alpha) in enumerate(cases()):
        if args.case and i+1 not in args.case:
            continue
        roles = evidence_partition(r, rotations=[LAW.T], seed=384+i)['roles']
        work, selection, audit = (r.subset(roles == role) for role in ('work', 'selection', 'audit'))
        elements = [a.element for a in truth.atoms]
        options = PhasingOptions(trials=args.trials, iterations=args.iterations, keep=min(3, args.trials),
                                 seed=853+i, algorithm='hybrid', initialization=args.initialization)
        methods = {}
        for mode in ('single_seed', 'joint_cpu', *([args.gpu] if args.gpu else [])):
            started = perf_counter()
            search = {}
            if mode == 'single_seed':
                candidates = solve_p1(truth.cell, work, elements, options).candidates
            else:
                search = direct_twin_phasing(truth.cell, work, elements, LAW, options, assume_friedel=True,
                                             backend='batched_cpu' if mode == 'joint_cpu' else mode,
                                             coarse_iterations=args.coarse_iterations, promote=min(3, args.trials),
                                             inspect_rejections=args.inspect_rejections)
                candidates = search['candidates']
            phasing_elapsed = perf_counter()-started
            metrics = score_candidates(candidates, work, selection, audit, truth, args.refinement_iterations)
            overhead = search.get('runtime', {}).get('inspection_seconds', 0.)
            methods[mode] = {**metrics, 'phase_seconds': phasing_elapsed-overhead,
                             'total_seconds': perf_counter()-started-overhead,
                             **search.get('runtime', {})}
            if search.get('audited_rejections'):
                methods[mode]['screened_recoveries'] = score_candidates(
                    search['audited_rejections'], work, selection, None, truth, args.refinement_iterations
                )['recoverable_candidates']
                methods[mode]['screened_rechecks'] = len(search['audited_rejections'])
        row = dict(case=f'T{i+1:02d}', fraction=alpha, work=len(work), selection=len(selection),
                   audit=len(audit), methods=methods)
        print(json.dumps(row, allow_nan=False), flush=True)
        cases_out.append(row)
    return {'schema': 'ariadne-twin-blind-comparison/1', 'cases': cases_out, 'recovery_cutoff_angstrom': RMS_CUTOFF,
            'trials': args.trials, 'iterations': args.iterations, 'coarse_iterations': args.coarse_iterations,
            'refinement_iterations': args.refinement_iterations, 'initialization': args.initialization,
            'scope': 'Synthetic signed intensities. All methods refine the same two-domain model after seeding.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('metal', 'cuda'))
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--coarse-iterations', type=int, default=55)
    parser.add_argument('--inspect-rejections', type=int, default=1)
    parser.add_argument('--refinement-iterations', type=int, default=80)
    parser.add_argument('--initialization', choices=('random', 'mixed', 'patterson'), default='mixed')
    parser.add_argument('--case', action='append', type=int)
    args = parser.parse_args()
    result = benchmark(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
