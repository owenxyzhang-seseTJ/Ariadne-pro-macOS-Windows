"""Plan bounded detector rereads using calibrated profile shapes and frame costs."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np


def preview_information(contributors):
    """Conditional Gaussian profile contrast after eliminating the common sum.

    The value describes unit-exposure equal-noise design geometry and is used
    only for prioritization. Pixel likelihoods use the detector's actual model.
    """
    if len(contributors) < 2:
        return 'isolated', 0., None
    distances, contrasts = [], []
    for first, second in combinations(contributors, 2):
        if first['domain'] == second['domain']:
            continue
        a = np.asarray(first['entry']['covariance_px_xyz'], dtype=float).reshape(3, 3)
        b = np.asarray(second['entry']['covariance_px_xyz'], dtype=float).reshape(3, 3)
        delta = np.asarray(first['centre'])-np.asarray(second['centre'])
        average = (a+b)*.5
        separation = float(delta @ np.linalg.solve(average, delta))
        # L2 overlap of two unit-amplitude normal profiles, normalized by
        # their respective norms. Width differences can also break overlap.
        log_correlation = (.25*(np.linalg.slogdet(a)[1]+np.linalg.slogdet(b)[1])
                           - .5*np.linalg.slogdet(average)[1] - separation/4.)
        correlation = float(np.exp(min(0., log_correlation)))
        distances.append(np.sqrt(max(0., separation)))
        contrasts.append(1.-correlation)
    if not distances:
        return 'same_domain_neighbours', 0., None
    return ('almost_coincident' if max(contrasts) < .01 else 'partial_overlap',
            float(max(contrasts)), float(max(distances)))


def choose_windows(windows, frame_budget, *, seed=1729, random_fraction=.15,
                   frame_seconds=.03, seconds_per_voxel=5e-7):
    """Choose complete disjoint patches. All contributors of a patch stay together."""
    if (type(frame_budget) is not int or frame_budget < 1 or
            not 0 <= random_fraction < .5 or frame_seconds <= 0 or seconds_per_voxel <= 0):
        raise ValueError('Invalid targeted-window budget or read-cost model')
    rng = np.random.default_rng(seed)
    frames, selected, reasons = set(), set(), {}
    scans = defaultdict(list)
    for i, window in enumerate(windows):
        scans[window['scan']].append(i)

    def add(i, cap, reason):
        new = windows[i]['frames'] - frames
        if len(frames)+len(new) > cap:
            return False
        selected.add(i)
        frames.update(new)
        reasons[i] = reason
        return True

    # Reserve independent random-window capacity by physical scan. It is
    # descriptive (selection-bias check), separate from external audit roles.
    reserve = min(frame_budget-1, max(1, round(frame_budget*random_fraction))) if random_fraction else 0
    quotas = max(1, reserve//max(1, len(scans)))
    for scan in sorted(scans) if reserve else ():
        stop = min(reserve, len(frames)+quotas)
        for i in rng.permutation(scans[scan]):
            if add(int(i), stop, 'random_scan_window'):
                break

    frames_to_windows = defaultdict(list)
    for i, window in enumerate(windows):
        for frame in window['frames']:
            frames_to_windows[frame].append(i)

    # Greedy with lazy updates when frame costs fall. A frame that has
    # already been selected costs only its additional patch extraction.
    from heapq import heappop, heappush
    import math

    utility = []
    for window in windows:
        kind = window['classification']
        value = (1.+window['information'] if kind == 'partial_overlap' else
                 .8 if kind == 'isolated' else .12 if kind == 'almost_coincident' else .25)
        # Independent minor-domain anchor windows and high-angle probes are
        # valuable even when there is no neighbouring primary-domain peak.
        if kind == 'isolated' and window['domain'] > 0:
            value += .5
        utility.append(value*(1.+.25*min(2., math.sqrt(max(0., window['reciprocal_sq'])))))

    def priority(i):
        window = windows[i]
        cost = frame_seconds*len(window['frames']-frames)+seconds_per_voxel*window['voxels']
        return -utility[i]/cost

    def spend(indices, cap):
        candidates = set(indices)-selected
        heap = []
        version = dict.fromkeys(candidates, 0)
        for i in sorted(candidates):
            heappush(heap, (priority(i), i, 0))
        while heap:
            _, i, stamp = heappop(heap)
            if i not in candidates or stamp != version[i]:
                continue
            new = windows[i]['frames']-frames
            if not add(i, cap, 'information_per_cost'):
                continue
            candidates.remove(i)
            for j in {j for frame in new for j in frames_to_windows[frame]} & candidates:
                version[j] += 1
                heappush(heap, (priority(j), j, version[j]))

    selected_per_scan = {}
    if scans:
        quota, remainder = divmod(frame_budget, len(scans))
        for position, (scan, indices) in enumerate(sorted(scans.items())):
            scan_frames = {frame for i in indices for frame in windows[i]['frames']}
            used = len(scan_frames & frames)
            allowance = max(0, quota+int(position < remainder)-used)
            spend(indices, min(frame_budget, len(frames)+allowance))
        spend(range(len(windows)), frame_budget)
        selected_per_scan = {str(scan): len({f for i in indices if i in selected for f in windows[i]['frames']})
                             for scan, indices in scans.items()}
    return selected, {'budget_distinct_frames': frame_budget,
                      'planned_distinct_frames': len(frames),
                      'available_distinct_frames': len(frames_to_windows),
                      'window_count': len(windows), 'selected_window_count': len(selected),
                      'seed': seed, 'reason_by_window': reasons, 'selected_frames_by_scan': selected_per_scan,
                      'cost': {'frame_seconds': frame_seconds, 'seconds_per_voxel': seconds_per_voxel}}
