"""Pair staged resolution and exhaustive refinement using identical starts and work rows."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from ariadne.agent import engine, resolve
from ariadne.agent.actions import Context
from ariadne.agent.store import Store
from ariadne.agent.util import read_json
from ariadne_xtal.io import write_reflections, write_structure
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import calculated_intensities
from benchmark_upgrade import peak_memory_mib


def case(root, index, alpha, cubic):
    random = np.random.default_rng(903+index)
    cell = Cell(10.2, 10.2, 10.2) if cubic else Cell(8.3, 9.2, 10.7)
    law = np.diag([1, -1, -1]).tolist()
    atoms = [Atom(f'{element}{i+1}', element, p, u_iso=.026)
             for i, (element, p) in enumerate(zip(['Fe', 'Cl', 'O', 'N', 'O'], random.random((5, 3))))]
    truth = Structure(cell, atoms)
    hkl = np.array(list(np.ndindex((15, 15, 15))))-7
    hkl = hkl[np.any(hkl, axis=1)]
    signal = calculated_intensities(truth, hkl, twin_laws=[law], twin_fractions=[1-alpha, alpha])
    sigma = np.hypot(4., .025*signal)
    r = Reflections(hkl, signal + random.normal(0., sigma), sigma)
    model = Structure.from_dict(truth.to_dict())
    for atom in model.atoms:
        atom.frac = (atom.frac + random.normal(0., .011, 3)) % 1
    folder = root/f'C{index+1:02d}'
    folder.mkdir()
    write_structure(model, folder/'prior.json')
    write_reflections(r, folder/'obs.json')
    request = dict(mode='resolve', inputs={'hkl': str(folder/'obs.json'), 'structure': str(folder/'prior.json')},
                   settings={'spacegroup_candidates': [], 'short_evaluations': 6, 'full_evaluations': 45,
                             'shortlist': 4 if cubic else 3, 'finalists': 2, 'inspect_rejections': int(cubic),
                             'refinement_options': {'reflection_block_size': 384}},
                   cache={'directory': str(folder/'cache'), 'max_mib': 64})
    session = folder/'session'
    engine.initialize(request, session)
    start = perf_counter()
    engine.run(session, max_steps=3)
    preparation_seconds = perf_counter()-start
    with closing(Store(session)) as store:
        start = perf_counter()
        state = engine.run(session)
        accelerated_seconds = perf_counter()-start
        failures = [n for n in state['nodes'] if n['status'] != 'succeeded']
        if failures:
            raise RuntimeError(str(failures))
        screen = store.node('screen')['result']
        report = read_json(session/screen['report']['path'])
        prep = store.node('evidence')['result']
        selection = store.node('select')['result']
        n_full = sum(n['id'].startswith('full-') for n in state['nodes'])
        n_short = sum(n['id'].startswith('short-') for n in state['nodes'])
        fits = []
        start = perf_counter()
        for row in report['rows']:
            if not row['eligible']:
                continue
            directory = session/'exhaustive'/row['id']
            directory.mkdir(parents=True)
            context = Context(store, 'screen', directory, perf_counter())
            fits.append(resolve.fit(context, dict(preparation=prep, candidate=row, stage='full')))
        exhaustive_seconds = perf_counter()-start
    warm = folder/'warm'
    engine.initialize(request, warm)
    start = perf_counter()
    engine.run(warm, max_steps=3)
    warm_seconds = perf_counter()-start
    with closing(Store(warm)) as store:
        warm_screen = store.node('screen')['result']['report']
        warm_rows = read_json(warm/warm_screen['path'])['rows']
        diagnostic = store.node('evidence')['result']['diagnostic']
        statistics = [r['batch_statistics'] for r in warm_rows if 'batch_statistics' in r]
        cache_hit = (bool(statistics) and all(s.get('stage_cache_hit') for s in statistics)
                     and read_json(warm/diagnostic['path'])['stage_cache_hit'])
    fitted = [row for row in fits if row['converged'] and row['identifiable']]
    best = min(fitted, key=lambda row: row['selection_mse'])
    preferred = selection.get('preferred')
    expected = np.asarray(law) if alpha else None
    expected_law = None if expected is None else law

    def correct(candidate):
        if candidate is None:
            return False
        actual = candidate['matrix']
        if expected_law is None or actual is None:
            return expected_law is actual
        return np.array_equal(actual, expected) or np.array_equal(actual, -expected)

    return dict(case=folder.name, known_fraction=alpha, metric='cubic' if cubic else 'orthorhombic',
                total_reflections=len(r), staged_selected_correct=correct(preferred),
                exhaustive_selected_correct=correct(best), staged_decision=selection['status'],
                exhaustive_full_fits=len(fits), staged_full_fits=n_full, staged_short_fits=n_short,
                shared_preparation_seconds=preparation_seconds,
                warm_preparation_seconds=warm_seconds, warm_stage_cache_hit=cache_hit,
                cached_structure_factor_evaluations=sum(s['evaluated_unique_hkl'] for s in statistics
                                                        if s.get('stage_cache_hit')),
                exhaustive_total_seconds=preparation_seconds+exhaustive_seconds,
                staged_total_seconds=preparation_seconds+accelerated_seconds,
                compared_identifiable=len(fitted))


def cache_trial(max_cache_mib):
    from ariadne_xtal.twin_diagnostics import twin_laws, twin_screen

    rng = np.random.default_rng(8172)
    cell = Cell(27., 27., 27.)
    elements = ['Fe', 'Cl', 'O', 'N', 'C'] * 3
    model = Structure(cell, [Atom(f'{element}{i+1}', element, rng.random(3), u_iso=.026)
                             for i, element in enumerate(elements)])
    hkl = rng.integers(-32, 33, size=(100_000, 3))
    hkl[~np.any(hkl, axis=1), 0] = 1
    laws = twin_laws(cell, 'P 1')[:16]
    signal = calculated_intensities(model, hkl, twin_laws=[laws[0]['matrix']], twin_fractions=[.65, .35])
    sigma = np.hypot(3., .025*signal)
    observations = Reflections(hkl, signal+rng.normal(0., sigma), sigma)
    start = perf_counter()
    screened = twin_screen(model, observations, laws, max_cache_mib=max_cache_mib)
    elapsed = perf_counter()-start
    best = min(screened['rows'], key=lambda row: row['chi_square'])
    return dict(schema='ariadne-twin-screen-memory/1', synthetic_reflections=len(observations),
                twin_laws=len(laws), elapsed_seconds=elapsed, peak_process_mib=peak_memory_mib(),
                competitive_chi_square=best['chi_square'], fraction=best['fraction'], cache=screened['cache'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New output directory')
    parser.add_argument('--batch-cache-mib', type=int, help='Measure one fixed 100,000-reflection memory-budget trial')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        if args.batch_cache_mib is not None:
            result = cache_trial(args.batch_cache_mib)
            (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print(json.dumps(result, allow_nan=False))
            return
        rows = []
        for i, (alpha, cubic) in enumerate(((0., False), (.05, True), (.2, True), (.45, True), (.5, True))):
            value = case(args.output, i, alpha, cubic)
            rows.append(value)
            print(json.dumps(value, allow_nan=False), flush=True)
    (args.output/'summary.json').write_text(json.dumps({'schema': 'ariadne-twin-resolution-comparison/1',
        'cases': rows, 'peak_single_process_mib': peak_memory_mib(),
        'scope': 'Both routes share the same snapshots, candidate generation and fixed-model warm starts. '
                                'The exhaustive route refines every family at the same full evaluation cap.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
