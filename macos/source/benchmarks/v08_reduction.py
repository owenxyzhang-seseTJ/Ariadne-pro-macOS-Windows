#!/usr/bin/env python3
"""Reproducible synthetic likelihood/observability and workspace measurements.

Run from the source root with an installed ariadne package:
    python benchmarks/v08_reduction.py
No raw diffraction dataset, DIALS execution, or clinical-style coverage claim
is implied by these controlled examples.
"""

import json
import os
from pathlib import Path
import platform
import time
import tracemalloc

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy
from scipy.linalg import qr

from ariadne.reduction_native import compress_gaussian, fit_poisson


def measure(function):
    tracemalloc.start()
    start = time.perf_counter()
    answer = function()
    elapsed = time.perf_counter()-start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return answer, {"seconds": elapsed, "peak_traced_bytes": peak}


def main():
    rng = np.random.default_rng(840108)
    summary = {
        "schema": "ariadne-v08-reduction-validation/1",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "machine": platform.machine()},
        "data_kind": "synthetic controlled examples; no raw DIALS experiment in this benchmark",
        "seed": 840108,
    }
    # Gaussian data with one large and one genuinely weak amplitude.
    x = np.linspace(-30, 30, 101)
    p = np.stack((np.exp(-.5*((x+5)/4)**2), np.exp(-.5*((x-5)/4)**2)), axis=1)
    p /= p.sum(axis=0)
    b = np.ones((len(x), 1))
    observed = p @ [500., 4.] + 5. + rng.normal(0, 2., len(x))
    compressed = compress_gaussian(observed, p, variance=4., background=b, chunk_rows=17)
    candidate_errors = []
    for intensities in rng.normal(0, 300, (100, 2)):
        residual = (observed-p@intensities)/2
        nuisance = np.linalg.lstsq(b/2, residual, rcond=None)[0]
        direct = float(np.sum((residual-b/2@nuisance)**2))
        candidate_errors.append(abs(compressed.chi2(intensities)-direct)/max(1., direct))
    summary["gaussian_strong_weak"] = {
        "truth": [500., 4.], "fit": compressed.fit(),
        "maximum_relative_compressed_likelihood_error": max(candidate_errors),
        "arbitrary_test_structural_predictions": 100,
    }

    # Exact profile reference: n1~Poisson(I+b), n2~Poisson(b).
    signal = np.array([[1.], [0.]])
    joint = fit_poisson([8, 3], signal, background=np.ones((2, 1)))
    frozen = fit_poisson([8, 3], signal, offset=np.full(2, 3.))
    summary["joint_count_background"] = {
        "data": [8, 3], "model": "mu=[I+b,b]",
        "fit": joint.to_dict(), "joint_background_interval": joint.profile_interval([1]),
        "frozen_background_interval": frozen.profile_interval([1]),
    }
    high_count, known_offset = 10**12, 10**7
    high = fit_poisson([high_count], [[1.]], offset=[known_offset])
    summary["high_count_known_offset"] = {
        "count": high_count, "known_offset": known_offset,
        "closed_form_mle": high_count-known_offset,
        "error_in_poisson_sigma": float((high.intensities_internal[0]-(high_count-known_offset))/np.sqrt(high_count)),
        "fit": high.to_dict(),
        "purpose": "regression for projected-gradient cancellation at large count coordinates",
    }

    full_overlap = np.array([[.2, .2], [.3, .3], [.5, .5]])
    twin = fit_poisson([3, 2, 5], full_overlap)
    summary["complete_twin_overlap"] = {
        "fit": twin.to_dict(), "sum_interval": twin.profile_interval([1, 1]),
        "individual_interval": twin.profile_interval([1, 0]),
    }

    overlap_series = []
    for separation in (0., .01, .05, .2, .5, 1., 2., 4., 8., 16., 24.):
        profiles = np.stack((np.exp(-.5*((x+separation/2)/4)**2),
                             np.exp(-.5*((x-separation/2)/4)**2)), axis=1)
        profiles /= profiles.sum(axis=0)
        observation = compress_gaussian(np.zeros(len(x)), profiles, variance=4., background=b)
        fit = observation.fit()
        overlap_series.append({
            "separation_pixels": separation, "profile_sigma_pixels": 4.,
            "rank": observation.rank,
            "information_ratio": fit['conditional_information_ratio'],
            "individual_variance": None if fit['covariance'] is None else np.diag(fit['covariance']).tolist(),
            "sum_variance": observation.contrast([1, 1])['variance'],
        })
    summary['overlap_observability_curve'] = overlap_series

    # An analytic clipping bias demonstration; these are repeated scalar
    # measurements, not a measured performance gain on a crystal dataset.
    signal_truth, sigma, repeats = 0., 3., 100000
    simulated = rng.normal(signal_truth, sigma, repeats)
    summary['weak_signal_sign_selection'] = {
        "truth": signal_truth, "sigma": sigma, "repetitions": repeats,
        "signed_estimate_mean": float(simulated.mean()),
        "clipped_estimate_mean": float(np.maximum(simulated, 0).mean()),
        "analytic_clipped_expectation": sigma/np.sqrt(2*np.pi),
        "negative_fraction": float(np.mean(simulated < 0)),
    }

    # Inputs are allocated before measurement. The measured allocation is
    # additional work memory, not total process RSS or caller-owned image data.
    n, k, chunk = 250000, 4, 1024
    xx = np.linspace(-1, 1, n)
    profiles = np.column_stack([np.exp(-40*(xx-center)**2) for center in (-.6, -.2, .2, .6)])
    profiles /= profiles.sum(axis=0)
    background = np.column_stack((np.ones(n), xx))
    y = profiles @ [4e4, 2e4, 1e4, 3e4] + background @ [2., .1] + rng.normal(size=n)
    compressed_large, streaming_stats = measure(lambda: compress_gaussian(
        y, profiles, variance=1., background=background, chunk_rows=chunk))

    def direct_augmented_qr():
        matrix = np.column_stack((background, profiles, y))
        return qr(matrix, mode='r', overwrite_a=True, check_finite=False)[0][:k+3].copy()

    dense_factor, dense_stats = measure(direct_augmented_qr)
    summary['workspace'] = {
        "pixels": n, "signal_columns": k, "background_columns": 2, "chunk_rows": chunk,
        "streamed_complete_compression": streaming_stats,
        "dense_augmented_qr_only": dense_stats,
        "peak_traced_memory_ratio_dense_over_streamed": dense_stats['peak_traced_bytes']/streaming_stats['peak_traced_bytes'],
        "note": "single measured run; dense baseline performs QR only, streamed path additionally profiles background and computes SVD diagnostics; tracemalloc peak excludes preallocated input arrays and is not process RSS",
        "output_rank": compressed_large.rank,
        "dense_factor_shape": list(dense_factor.shape),
    }
    destination = Path(__file__).with_name('v08_reduction_results.json')
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"output": str(destination), "pixel_likelihood_relative_error": max(candidate_errors),
                      "joint_interval": summary['joint_count_background']['joint_background_interval'],
                      "workspace": summary['workspace']}, indent=2))


if __name__ == '__main__':
    main()
