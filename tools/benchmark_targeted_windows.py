"""Paired detector-window selection benchmark with an independent pixel scorer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


def patch_scores(path, window):
    with np.load(path/window['patch'], allow_pickle=False) as patch:
        valid = patch['valid'].ravel()
        y = patch['data'].ravel()[valid]
        variance = patch['variance'].ravel()[valid]
        profiles = patch['profiles'].reshape(len(window['contributors']), -1).T[valid]
    if len(y) < 12:
        return None
    # Interleaved calibration/scoring voxels. Each model uses precisely the
    # same pixels and weights. All profile columns come from fixed geometry.
    work = np.flatnonzero(valid) % 2 == 0
    if work.sum() <= profiles.shape[1]+1 or not np.any(~work):
        return None
    root = np.sqrt(variance)
    intercept = np.ones((len(y), 1))
    design = np.concatenate([intercept, profiles], axis=1)/root[:, None]
    standardized = y/root
    columns_for_domain0 = [0, *[i+1 for i, contributor in enumerate(window['contributors'])
                                if contributor['domain'] == 0]]
    losses = {}
    for label, columns in (('background_loss', design[:, :1]), ('primary_loss', design[:, columns_for_domain0]),
                           ('peak_loss', design)):
        fitted = np.linalg.lstsq(columns[work], standardized[work], rcond=None)[0]
        residual = columns[~work] @ fitted - standardized[~work]
        losses[label] = float(residual @ residual)
    return dict(pixels=int(np.sum(~work)), **losses)


def trial(prepare, args, output, frame_limit):
    start = perf_counter()
    prepare(SimpleNamespace(expt=args.experiments, predictions=args.predictions, library=args.profiles,
                            output=output, support_sigma=args.support_sigma, background_margin=2,
                            frame_cache=32, frame_cache_mib=args.cache_mib, template_cache=32,
                            template_cache_mib=128, position_derivatives=False, max_voxels=250_000,
                            max_components=16, target_frames=frame_limit, selection_seed=args.seed,
                            geometry_diagnostics=True))
    elapsed = perf_counter()-start
    manifest = json.loads((output/'patches.json').read_text())
    scores, summaries, overlaps = {}, defaultdict(Counter), defaultdict(Counter)
    start = perf_counter()
    for entry in manifest['groups']:
        geometry = entry['geometry_selection']
        value = patch_scores(output, entry)
        key = entry['observation_id']
        row = summaries[geometry['scan']]
        row['windows'] += 1
        row[geometry['classification']] += 1
        if value:
            row.update(value)
            scores[key] = value
            if geometry['classification'] == 'partial_overlap':
                overlap = overlaps[geometry['scan']]
                overlap.update(value)
                overlap['windows'] += 1
                overlap['all_domains_better'] += value['peak_loss'] < value['primary_loss']
    return {'elapsed_seconds': elapsed, 'score_seconds': perf_counter()-start,
            'windows': len(manifest['groups']), 'frame_cache': manifest['cache']['frames'],
            'targeted_selection': {k: v for k,v in (manifest['targeted_selection'] or {}).items()
                                   if k != 'reason_by_window'},
            'window_scores_by_scan': dict(summaries), 'partial_overlap_scores_by_scan': dict(overlaps)}, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiments', 'predictions', 'profiles', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--target-frames', type=int, required=True)
    parser.add_argument('--cache-mib', type=int, default=384)
    parser.add_argument('--support-sigma', type=float, default=3.)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--full-first', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'macos/source/skills/ariadne-reduction/scripts'))
    from overlap_prepare import prepare

    args.output.mkdir(parents=True, exist_ok=False)
    plan = [('targeted', args.target_frames), ('full', None)]
    if args.full_first:
        plan.reverse()
    metrics, scores = {}, {}
    for kind, cap in plan:
        metrics[kind], scores[kind] = trial(prepare, args, args.output/kind, cap)
    shared = scores['targeted'].keys() & scores['full'].keys()
    metrics['common_window_scores_agree'] = len(shared) == len(scores['targeted']) and all(
        scores['targeted'][key] == scores['full'][key] for key in shared)
    metrics['scored common windows'] = len(shared)
    metrics['scope'] = ('Fixed, supplied detector geometry and model predictions. Independent interleaved '
                        'pixel scoring conditions on those predictions and on supplied noise weights.')
    (args.output/'summary.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
