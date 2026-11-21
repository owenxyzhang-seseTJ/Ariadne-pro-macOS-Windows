"""Compare CPU/GPU phase screening at equal trial budgets and redact the outputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter

import gemmi
import numpy as np
from scipy.optimize import linear_sum_assignment

from ariadne_xtal.io import read_structure
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.phasing import PhasingOptions
from ariadne_xtal.scattering import calculated_intensities
from ariadne_xtal.screened_phasing import screened_phase_search
from ariadne_xtal.symmetry import unique_sites


RECOVERY_CUTOFF = .25


@dataclass
class Case:
    code: str
    source: str
    reference: Structure
    search: Reflections
    measured: Reflections


def reciprocal_orbit(hkl, operators):
    equivalents = {tuple(op.apply_to_hkl(hkl)) for op in operators} if operators else {tuple(hkl)}
    return {min(h, tuple(-v for v in h)) for h in equivalents}


def holdout_flags(hkl, seed, operators=()):
    keys = [min(reciprocal_orbit(h, operators)) for h in hkl.tolist()]
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    picked = np.zeros(len(unique), dtype=bool)
    picked[np.random.default_rng(seed).permutation(len(unique))[::5]] = True
    return picked[inverse]


def synthetic_cases():
    # Cases cover two diffraction grid scales with the same supplied composition.
    for index, length in enumerate([5.4, 6.2, 6.6, 10.7, 12.3, 14.1]):
        rng = np.random.default_rng(7810 + index)
        cell = Cell(length, length+.37, length+.82)
        positions = []
        while len(positions) < 4:
            x = rng.uniform(0.,1.,3)
            if all(np.linalg.norm(cell.to_cartesian(cell.minimum_image(x-y))) > 1.1 for y in positions):
                positions.append(x)
        truth = Structure(cell, [Atom(f'C{i+1}', 'C', value, u_iso=.012)
                                  for i,value in enumerate(positions)])
        cutoff = .68
        limits = [int(np.ceil(edge/cutoff)) for edge in (cell.a,cell.b,cell.c)]
        all_h = np.array(list(product(*(range(-n,n+1) for n in limits))))
        all_h = all_h[np.any(all_h,axis=1)]
        hkl = all_h[cell.d_spacing(all_h) >= cutoff]
        signal = calculated_intensities(truth,hkl)
        sigma = np.hypot(.5,.045*signal)
        observed = signal + rng.normal(size=len(hkl))*sigma
        observations = Reflections(hkl, observed, sigma, free_flags=holdout_flags(hkl, 9414+index))
        yield Case(f'S{index+1:02d}', 'synthetic', truth, observations, observations)


def measured_cases(structure_path, intensities_path):
    reference = read_structure(structure_path)
    atoms = [Atom(f'{atom.element}{i+1}',atom.element,xyz,u_iso=.02)
             for i,(atom,xyz) in enumerate((atom,xyz) for atom in reference.atoms
                                           for xyz,_ in unique_sites(reference,atom))]
    truth = Structure(reference.cell,atoms,spacegroup='P 1')
    block = gemmi.cif.read_file(str(intensities_path)).sole_block()
    columns = ['index_h','index_k','index_l','F_squared_meas','F_squared_sigma']
    rows = np.array([[gemmi.cif.as_number(value) for value in row]
                     for row in block.find('_refln_',columns)])
    if rows.ndim != 2 or len(rows) < 20:
        raise ValueError('Provide a structure-factor CIF with measured intensities and uncertainties')
    if not np.isfinite(rows).all() or np.any(rows[:, :3] != np.rint(rows[:, :3])):
        raise ValueError('The intensity table has nonnumeric measurements or nonintegral indices')
    hkl = rows[:,:3].astype(int)
    operators = tuple(reference.operations)
    flags = holdout_flags(hkl, 912, operators)
    measured = Reflections(hkl, rows[:, 3], rows[:, 4], free_flags=flags)
    yield Case('M01', 'measured_as_recorded', truth, measured, measured)
    # Expand only work orbits to supply P1 reciprocal coverage. Repeated
    # representatives of an orbit are pooled before expansion.
    groups = {}
    for h, y, sigma in zip(measured.hkl[~flags], measured.intensity[~flags], measured.sigma[~flags]):
        orbit = reciprocal_orbit(h.tolist(), operators)
        groups.setdefault(tuple(sorted(orbit)), []).append((y, sigma))
    original = {next(iter(reciprocal_orbit(h, ()))) for h in measured.hkl[~flags].tolist()}
    if {h for orbit in groups for h in orbit} <= original:
        return
    expanded = []
    for orbit, readings in groups.items():
        values, sigmas = np.asarray(readings).T
        smallest = min(sigmas)
        weights = (smallest/sigmas)**2
        mean = np.average(values, weights=weights)
        # Distribute the orbit weight across its copies for work scoring.
        sigma = smallest * np.sqrt(len(orbit)/weights.sum())
        expanded.extend((h, mean, sigma) for h in orbit)
    work_hkl, intensity, sigma = zip(*expanded)
    yield Case('M02', 'same_measurement_symmetry_expanded_work', truth,
               Reflections(work_hkl, intensity, sigma), measured)


def matched_rms(truth, candidate):
    atoms = [a.element for a in truth.atoms]
    acquired = [a.element for a in candidate.atoms]
    if sorted(atoms) != sorted(acquired):
        return None
    a = np.array([atom.frac for atom in truth.atoms])
    b = np.array([atom.frac for atom in candidate.atoms])
    species = np.equal(np.asarray(atoms)[:,None],np.asarray(acquired)[None,:])
    best = np.inf
    for sign in (-1,1):
        for index in np.flatnonzero(species[0]):
            offset = a[0]-sign*b[index]
            for _ in range(4):
                dx = truth.cell.minimum_image(a[:,None,:] - sign*b[None,:,:]-offset)
                cost = np.sum(truth.cell.to_cartesian(dx)**2,axis=2)
                i,j = linear_sum_assignment(np.where(species,cost,np.inf))
                offset += dx[i,j].mean(axis=0)
            residual = truth.cell.minimum_image(a[i]-sign*b[j]-offset)
            best = min(best,np.sqrt(np.mean(np.sum(truth.cell.to_cartesian(residual)**2,axis=1))))
    return float(best)


def heldout_rmse(model, obs):
    predictions = calculated_intensities(model,obs.hkl)
    work = ~obs.free_flags
    p = predictions / obs.sigma
    y = obs.intensity / obs.sigma
    scale = max(0.,float(p[work]@y[work]/(p[work]@p[work])))
    return float(np.sqrt(np.mean((scale*p[~work]-y[~work])**2)))


def run_case(case, backend, options, coarse_iterations, promoted, inspected):
    elements = [atom.element for atom in case.reference.atoms]
    started = perf_counter()
    result = screened_phase_search(case.reference.cell, case.search, elements, options,
                                   coarse_iterations=coarse_iterations,promoted=promoted,
                                   random_promotions=1,inspected_rejections=inspected,
                                   coarse_backend=backend)
    elapsed = perf_counter()-started
    # Coordinates and held-out readings are consumed only after the shortlist is fixed.
    matches = {value['trial']: matched_rms(case.reference,value['structure'])
               for value in [*result['candidates'],*result['audited_rejections']]}
    first = None
    clock = result['screening_runtime']['wall_seconds']
    approved = [trial for trial in result['trial_diagnostics'] if trial['promoted']]
    for trial in approved:
        clock += trial['wall_seconds']
        rms = matches.get(trial['trial'])
        if rms is not None and rms <= RECOVERY_CUTOFF and first is None:
            first = clock
    top = result['candidates'][0] if result['candidates'] else None
    return dict(case=case.code, source=case.source, backend=backend,
                input_rows=len(case.measured), search_rows=result['full_work_reflections'],
                audit_rows=int(case.measured.free_flags.sum()),
                coarse=result['screening_runtime'], wall_seconds=elapsed,
                search_seconds=clock,
                full_cpu_seconds=result['full_recheck_wall_seconds'],
                promoted_trials=[row['trial'] for row in approved],
                recovered_promotions=sum(matches[row['trial']] is not None
                                         and matches[row['trial']] <= RECOVERY_CUTOFF
                                         for row in approved if row['trial'] in matches),
                inspected_rejections=sum(not row['promoted'] for row in result['trial_diagnostics']),
                recovered_rejections=sum(matches[row['trial']] is not None
                                           and matches[row['trial']] <= RECOVERY_CUTOFF
                                           for row in result['trial_diagnostics']
                                           if not row['promoted'] and row['trial'] in matches),
                any_recovered=first is not None, first_recovery_seconds=first,
                top_rms_angstrom=matches.get(top['trial']) if top else None,
                top_heldout_sigma_rmse=heldout_rmse(top['structure'],case.measured) if top else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measured-structure',type=Path)
    parser.add_argument('--measured-intensities',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',choices=['metal','cuda'],default='metal')
    parser.add_argument('--trials',type=int,default=12)
    parser.add_argument('--coarse-iterations',type=int,default=55)
    parser.add_argument('--full-iterations',type=int,default=180)
    parser.add_argument('--promoted',type=int,default=3)
    parser.add_argument('--inspected',type=int,default=2)
    args=parser.parse_args()
    if bool(args.measured_structure) != bool(args.measured_intensities):
        parser.error('Provide both measured inputs')
    if args.gpu == 'metal':
        from mlx import core as mx

        if not mx.metal.is_available():
            raise RuntimeError('The paired benchmark requires a live Metal device')
        with mx.stream(mx.gpu):
            mx.eval(mx.fft.fftn(mx.ones((8,8,8))))
        device = dict(backend='metal', version=mx.__version__, device=str(mx.gpu))
    else:
        import cupy as cp

        cp.cuda.Device().use()
        cp.fft.fftn(cp.ones((8,8,8),dtype=cp.float32))
        cp.cuda.get_current_stream().synchronize()
        device = dict(backend='cuda', version=cp.__version__,
                      device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode())
    options=PhasingOptions(algorithm='hybrid',trials=args.trials,iterations=args.full_iterations,
                           keep=args.promoted,seed=1729)
    cases=list(synthetic_cases())
    if args.measured_structure:
        cases.extend(measured_cases(args.measured_structure,args.measured_intensities))
    measured=[]
    for index,case in enumerate(cases):
        backends=['batched_cpu',args.gpu] if index % 2 == 0 else [args.gpu,'batched_cpu']
        for backend in backends:
            row=run_case(case,backend,options,args.coarse_iterations,args.promoted,args.inspected)
            measured.append(row)
            print(row['case'],backend,round(row['coarse']['wall_seconds'],3),
                  round(row['search_seconds'],3),row['any_recovered'],flush=True)
    output=dict(schema='ariadne-blind-phasing-audit/1', gpu_device=device,
                cpu_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                trial_budget=dict(trials=args.trials, coarse_iterations=args.coarse_iterations,
                                  full_iterations=args.full_iterations, promoted=args.promoted,
                                  random_promotions=1, inspected=args.inspected),
                holdout='every fifth shuffled symmetry/Friedel group; scale fitted on original work observations',
                success_rms_angstrom=RECOVERY_CUTOFF, results=measured)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
