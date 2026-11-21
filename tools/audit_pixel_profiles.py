"""Audit pixel-profile transfer across full scans of calibrated photon counts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.spatial import cKDTree
from scipy.special import erf, gammaln, xlogy

from ariadne_xtal.pixel_profiles import GaussianPixels


@dataclass
class PixelCohort:
    counts: np.ndarray
    centres: np.ndarray
    widths: np.ndarray
    background: np.ndarray
    valid: np.ndarray
    scan: np.ndarray
    frame: np.ndarray
    side: int

    def profile(self, indices):
        y, x = np.indices((self.side, self.side))
        lower = np.column_stack([x.ravel(), y.ravel()])
        return GaussianPixels(lower, lower+1., self.centres[indices], self.widths[indices])


def _amplitude(counts, background, profile):
    """Nonnegative scalar Poisson fit with fixed background, separately per row."""
    total = profile.sum(axis=1)
    low = np.zeros(len(profile))
    high = np.divide(counts.sum(axis=1), total, out=low.copy(), where=total > 0.)

    def score(value):
        return np.sum(profile * (1. - counts/(background + value[:, None]*profile)), axis=1)

    boundary = score(low) >= 0.
    for _ in range(36):
        middle = (low+high)*.5
        left = (score(middle) < 0.) & ~boundary
        low = np.where(left, middle, low)
        high = np.where(left, high, middle)
    return np.where(boundary, 0., (low+high)*.5)


def _reference_losses(cohort, indices, parameters, calibration, audit):
    """Score unseen pixels with separate Gaussian integration and scalar fitting."""
    centres = cohort.centres[indices] + parameters[:2]
    widths = cohort.widths[indices] * np.exp(parameters[2])
    edges = (np.arange(cohort.side + 1)[None, :, None] - centres[:, None, :])
    mass = .5 * np.diff(erf(edges / (np.sqrt(2.) * widths[:, None, :])), axis=1)
    profiles = (mass[:, :, 1, None] * mass[:, None, :, 0]).reshape(len(indices), -1)
    losses = []
    for profile, counts, background, valid in zip(
            profiles, cohort.counts[indices], cohort.background[indices], cohort.valid[indices]):
        fit, score = valid & calibration, valid & audit
        p, y = profile[fit], counts[fit]

        def derivative(amplitude):
            return p.sum() - np.dot(p, y / (background + amplitude*p))

        amplitude = 0.
        if derivative(0.) < 0.:
            upper = y.sum() / p.sum()
            amplitude = brentq(derivative, 0., upper) if derivative(upper) > 0. else upper
        expected = background + amplitude*profile[score]
        observed = counts[score]
        losses.append(np.sum(expected - xlogy(observed, expected) + gammaln(observed + 1.)))
    return np.asarray(losses)


def evaluate(cohort, output_seed=831, *, max_width_ratio=2.):
    """Choose neither masks nor shape parameters using the held-out pixels."""
    yy, xx = np.indices((cohort.side, cohort.side))
    core = ((xx-cohort.side//2)**2 + (yy-cohort.side//2)**2 <= 5**2).ravel()
    calibration = ((xx+yy) % 2 == 0).ravel() & core
    audit = core & ~calibration
    scans = np.unique(cohort.scan)
    if len(scans) < 3:
        raise ValueError('At least three complete scan cohorts are required')
    outcomes = []
    deltas = []
    rng = np.random.default_rng(output_seed)
    for test_scan in scans:
        work = np.flatnonzero(cohort.scan != test_scan)
        test = np.flatnonzero(cohort.scan == test_scan)
        if len(test) < 20:
            raise ValueError('An audit scan has fewer than twenty eligible isolated peaks')
        reference = cohort.profile(work)
        observed = cohort.counts[work]
        mask = cohort.valid[work] & core
        b = cohort.background[work, None]

        def objective(theta, width_only):
            parameters = np.r_[0., 0., theta] if width_only else theta
            pixels, derivative = reference.evaluate(parameters, derivatives=True)
            profile = np.where(mask, pixels.T, 0.)
            amplitude = _amplitude(np.where(mask, observed, 0.), b, profile)
            expected = b + amplitude[:, None]*profile
            residual = np.where(mask, 1. - observed/expected, 0.)
            half_deviance = expected-observed+xlogy(observed, observed/expected)
            gradient = np.einsum('rp,r,prd->d', residual, amplitude, derivative)
            gradient = gradient[2:] if width_only else gradient
            return float(np.sum(half_deviance[mask])), gradient

        started = perf_counter()
        width = np.log(max_width_ratio)
        fits = [minimize(objective, np.zeros(n), args=(n == 1,), method='L-BFGS-B',
                         jac=True, bounds=([(-width,width)] if n == 1 else
                                           [(-1.5,1.5),(-1.5,1.5),(-width,width)]),
                         options={'maxiter': 60, 'ftol': 1e-10}) for n in (1,3)]
        if not all(fit.success for fit in fits):
            raise RuntimeError('A work-only profile optimizer did not converge')
        fitted = [np.zeros(3), np.r_[0.,0.,fits[0].x], fits[1].x]
        parameters_used = ['fixed', 'width', 'shift_and_width']
        usable_calibration = cohort.valid[test] & calibration
        usable_audit = cohort.valid[test] & audit
        y = cohort.counts[test]
        row_scores = {name: _reference_losses(cohort, test, parameters, calibration, audit)
                      for name, parameters in zip(parameters_used, fitted)}
        delta = row_scores['fixed'] - row_scores['shift_and_width']
        width_delta = row_scores['fixed'] - row_scores['width']
        shift_delta = row_scores['width'] - row_scores['shift_and_width']
        audit_pixels = usable_audit.sum(axis=1)
        calibrating_counts = np.sum(np.where(usable_calibration,
                                             y-cohort.background[test,None],0.),axis=1)
        blocks = cohort.frame[test] // 15
        gains = np.column_stack([width_delta, shift_delta, delta, audit_pixels])
        block_data = np.array([gains[blocks == k].sum(axis=0) for k in np.unique(blocks)])
        deltas.append(block_data)
        outcomes.append(dict(case=f'A{len(outcomes)+1}', work_peaks=len(work),
                             audit_peaks=len(test), audit_pixels=int(audit_pixels.sum()),
                             nll={key: float(scores.sum()) for key,scores in row_scores.items()},
                             shifts_pixels=fitted[-1][:2].tolist(),
                             width_ratio=float(np.exp(fitted[-1][2])),
                             width_only_ratio=float(np.exp(fitted[1][2])),
                             improved_peaks=int(np.sum(delta > 0)),
                             width_only_improved_peaks=int(np.sum(width_delta > 0)),
                             median_nll_gain_per_pixel=float(np.median(delta/audit_pixels)),
                             intensity_strata=[{
                                 'band': name, 'peaks': int(selected.sum()),
                                 'width_gain_per_pixel': (float(width_delta[selected].sum()/audit_pixels[selected].sum())
                                                          if selected.any() else None),
                             } for name, selected in (('below_200_calibration_counts', calibrating_counts < 200),
                                                        ('at_least_200_calibration_counts', calibrating_counts >= 200))],
                             wall_seconds=perf_counter()-started))
    samples = sum(blocks[rng.integers(0,len(blocks),size=(2000,len(blocks)))].sum(axis=1)
                  for blocks in deltas)
    total = sum(blocks.sum(axis=0) for blocks in deltas)
    comparisons = {}
    for column, name in enumerate(('width_vs_fixed', 'shift_vs_width', 'shift_width_vs_fixed')):
        comparisons[name] = dict(
            mean_nll_gain_per_pixel=float(total[column]/total[-1]),
            conditional_block_interval_95=np.quantile(samples[:, column]/samples[:, -1],
                                                      [.025, .975]).tolist())
    return dict(schema='ariadne-pixel-profile-audit/2', folds=outcomes,
                total_peaks=int(len(cohort.scan)), counted_audit_pixels=int(total[-1]),
                comparisons=comparisons,
                block_size_frames=15,
                max_width_ratio=max_width_ratio,
                conditioning='fixed experiment geometry; isolated detector-plane peak windows',
                amplitude_partition='audit windows use even-parity core pixels for peak amplitude',
                background_partition='disjoint annular pixels with fixed half-count smoothing',
                audit_partition='whole scan excluded from shape fitting; odd-parity core pixels scored',
                score_implementation='independent erf rectangle masses, Brent amplitude fit and Poisson log score')


def load_cbf_cohort(experiments_path, reflections_path, neighbour_experiments, neighbours_path, *,
                    limit_per_frame=4, nominal_width=1.6):
    """Read calibrated PAD counts and construct non-overlapping peak windows."""
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    primary = flex.reflection_table.from_file(str(reflections_path))
    neighbours = flex.reflection_table.from_file(str(neighbours_path))
    experiments = ExperimentListFactory.from_json_file(str(experiments_path), check_format=True)
    xyz = primary['xyzcal.px'].as_numpy_array()
    ids = primary['id'].as_numpy_array()
    partiality = primary['partiality'].as_numpy_array()
    nxyz = neighbours['xyzcal.px'].as_numpy_array()
    nids = neighbours['id'].as_numpy_array()
    alternatives = ExperimentListFactory.from_json_file(str(neighbour_experiments), check_format=False)

    def identity(exp):
        # Several crystals can share the same raw rotation sequence.
        return exp.imageset.paths()[0], tuple(exp.imageset.get_array_range())

    labels = {identity(exp): scan for scan,exp in enumerate(experiments)}
    nscan = np.full(len(neighbours), -1)
    for index,exp in enumerate(alternatives):
        nscan[nids == index] = labels.get(identity(exp), -1)
    radius = 9
    side = 2*radius+1
    yy, xx = np.indices((side,side))
    radius2 = (xx-radius)**2+(yy-radius)**2
    annulus = (radius2 >= 7**2) & (radius2 <= 9**2)
    calibration = ((xx+yy) % 2 == 0) & (radius2 <= 5**2)
    records = []
    for scan, experiment in enumerate(experiments):
        if len(experiment.detector) != 1:
            raise ValueError('This calibrated audit loader expects a single photon-counting panel')
        panel = experiment.detector[0]
        if panel.get_type() != 'SENSOR_PAD' or panel.get_gain() != 1.:
            raise ValueError('Photon-count audit requires a unit-gain calibrated PAD')
        safe_low, saturated = panel.get_trusted_range()
        frame_begin, frame_end = experiment.imageset.get_array_range()
        wanted = (ids == scan) & (partiality >= .75) & np.all(np.isfinite(xyz),axis=1)
        predicted = xyz[wanted]
        nearby = nxyz[(nscan == scan) & np.all(np.isfinite(nxyz),axis=1)]
        if len(nearby) < 2:
            raise ValueError('Provide a prediction catalogue for every scan')
        metric = [1.,1.,10.]
        neighbours_two = cKDTree(nearby*metric).query(predicted*metric,k=2)[0]
        clear = predicted[(neighbours_two[:,0] < 2.) & (neighbours_two[:,1] > 2*radius+2)]
        frames = np.floor(clear[:,2]).astype(int)
        for frame in np.unique(frames):
            if not frame_begin <= frame < frame_end:
                continue
            candidates = clear[frames == frame, :2]
            # The sweep order and candidate order depend only on geometry.
            candidates = candidates[np.lexsort((candidates[:,1],candidates[:,0]))]
            raw = experiment.imageset.get_raw_data(int(frame-frame_begin))[0].as_numpy_array()
            good = experiment.imageset.get_mask(int(frame-frame_begin))[0].as_numpy_array()
            if not np.issubdtype(raw.dtype, np.integer):
                raise ValueError('The detector reader did not return integer photon counts')
            used = []
            for centre in candidates:
                x, y = np.floor(centre).astype(int)-radius
                if x < 0 or y < 0 or x+side > raw.shape[1] or y+side > raw.shape[0]:
                    continue
                if any(np.linalg.norm(centre-previous) < 2*radius+2 for previous in used):
                    continue
                view = raw[y:y+side,x:x+side]
                valid = good[y:y+side,x:x+side] & (view >= max(safe_low,0.)) & (view < saturated)
                sideband = valid & annulus
                signal = valid & calibration
                if (sideband.sum() < 40 or signal.sum() < 32
                        or not (valid & (radius2 <= 5**2) & ~calibration).any()):
                    continue
                background = (view[sideband].sum()+.5)/(sideband.sum()+.5)
                if view[signal].sum() - signal.sum()*background < 45.:
                    continue
                used.append(centre)
                records.append((np.where(valid,view,0).ravel(), centre-[x,y],
                                [nominal_width, nominal_width], background, valid.ravel(), scan, frame))
                if len(used) >= limit_per_frame:
                    break
    if not records:
        raise ValueError('No calibrated peak windows passed the eligibility criteria')
    return PixelCohort(*(np.asarray(column) for column in zip(*records)), side=side)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', type=Path, help='Calibrated EXPT with fixed detector geometry')
    parser.add_argument('reflections', type=Path, help='Primary-lattice DIALS prediction table')
    parser.add_argument('neighbour_experiments', type=Path, help='EXPT for the complete competing prediction catalogue')
    parser.add_argument('neighbours', type=Path, help='DIALS prediction table covering all competing lattices')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--peaks-per-frame', type=int, default=2)
    parser.add_argument('--nominal-width', type=float, default=1.6)
    parser.add_argument('--max-width-ratio', type=float, default=2.)
    args = parser.parse_args()
    if args.peaks_per_frame < 1 or not np.isfinite(args.nominal_width) or args.nominal_width <= 0.:
        parser.error('Specify a positive per-frame limit and Gaussian width')
    if not np.isfinite(args.max_width_ratio) or args.max_width_ratio <= 1.:
        parser.error('The width-ratio bound must exceed one')
    started = perf_counter()
    cohort = load_cbf_cohort(args.experiments, args.reflections, args.neighbour_experiments, args.neighbours,
                             limit_per_frame=args.peaks_per_frame, nominal_width=args.nominal_width)
    preparation = perf_counter()-started
    result = evaluate(cohort, max_width_ratio=args.max_width_ratio)
    result.update(nominal_width_pixels=args.nominal_width, cohort_preparation_seconds=preparation)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
