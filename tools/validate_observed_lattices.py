"""Evaluate candidate lattice addition on a reserved portion of a DIALS spot table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ariadne.lattice_selection import select_lattices


def evaluate(experiments_path, reflections_path, block_frames):
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(str(experiments_path), check_format=False)
    reflections = flex.reflection_table.from_file(str(reflections_path))
    required = {'rlp', 'miller_index', 'id', 'xyzobs.px.value'}
    missing = required - set(reflections.keys())
    if missing:
        raise ValueError(f'Missing DIALS reflection columns: {sorted(missing)}')
    q = np.asarray(reflections['rlp'])
    h = np.asarray(reflections['miller_index'])
    ids = np.asarray(reflections['id'])
    z = np.asarray(reflections['xyzobs.px.value'])[:, 2]
    finite = np.isfinite(q).all(axis=1) & np.isfinite(z)
    scans = {}
    for i, experiment in enumerate(experiments):
        paths = experiment.imageset.paths()
        key = (str(paths[0]), str(paths[-1]), len(paths))
        scans.setdefault(key, []).append(i)
    certificates = []
    for scan_number, (_, exp_ids) in enumerate(scans.items(), 1):
        within = finite & np.isin(ids, exp_ids)
        if within.sum() < 20 or len(exp_ids) < 2:
            continue
        scan_blocks = np.full(len(z), -1, dtype=int)
        scan_blocks[within] = np.floor((z[within] - z[within].min()) / block_frames).astype(int)
        possible = np.unique(scan_blocks[within])
        if len(possible) < 2:
            continue
        selection = within & np.isin(scan_blocks, possible[1::5])
        candidates, deviations, reference = [], [], []
        for exp_id in exp_ids:
            known = within & (ids == exp_id) & np.any(h != 0, axis=1)
            work = known & ~selection
            if np.count_nonzero(work) < 8 or np.linalg.matrix_rank(h[work]) < 3:
                continue
            bases = np.linalg.lstsq(h[work], q[work], rcond=None)[0]
            candidates.append(bases.T)
            deviations.extend(np.linalg.norm(q[work] - h[work] @ bases, axis=1))
            reference.append({'experiment_id': exp_id, 'work_rows': int(work.sum()),
                              'selection_rows_with_reference_index': int((known & selection).sum())})
        if len(candidates) < 2:
            continue
        sigma = float(np.median(deviations) / 1.53817)
        result = select_lattices(q[within], candidates, selection[within], sigma=sigma)
        certificates.append({**result, 'scan': scan_number, 'work_spots': int((within & ~selection).sum()),
                             'selection_spots': int(selection.sum()), 'work_noise_sigma_inv_angstrom': sigma,
                             'reference_candidates': reference})
    return {'source_spots': len(reflections), 'scan_block_frames': block_frames, 'evaluated_scans': certificates,
            'scope': 'Conditional selection per physical scan on previously indexed spots. Work-only basis and noise fit.'}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('experiments', type=Path)
    cli.add_argument('reflections', type=Path)
    cli.add_argument('--block-frames', type=int, default=20)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.block_frames < 1:
        cli.error('block-frames must be positive')
    result = evaluate(args.experiments, args.reflections, args.block_frames)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
