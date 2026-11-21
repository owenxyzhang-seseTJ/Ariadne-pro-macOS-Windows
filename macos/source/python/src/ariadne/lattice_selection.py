"""Experimental predictive selection of candidate reciprocal lattices."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .lattice_math import affine_integer_rank, periodic_gaussian, reduced_basis


def _fit_weights(ratios):
    weights = np.full(ratios.shape[1], 1 / ratios.shape[1])
    for _ in range(200):
        values = ratios + np.log(np.maximum(weights, 1e-15))
        updated = np.exp(values - logsumexp(values, axis=1, keepdims=True)).mean(axis=0)
        if np.max(np.abs(updated - weights)) < 1e-10:
            return updated
        weights = updated
    return weights


def select_lattices(q, candidate_bases, selection_mask, *, sigma, maximum_lattices=3,
                    minimum_gain=10., minimum_support=6):
    """Add lattices proposed from work data only when a reserved scan supports them.

    Candidate bases have reciprocal column vectors in inverse angstrom.
    The wrapped Gaussian and uniform terms use a common reference volume.
    Weights describe centroid membership, with a uniform outlier component.
    """
    q = np.asarray(q, float)
    mask = np.asarray(selection_mask)
    if (q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or mask.dtype != np.bool_
            or mask.shape != q.shape[:1] or not mask.any() or mask.all()):
        raise ValueError("Supply finite reciprocal observations with nonempty work and selection scans")
    if (not np.isfinite([sigma, minimum_gain]).all() or sigma <= 0 or minimum_gain <= 0
            or type(maximum_lattices) is not int or not 1 <= maximum_lattices <= 16
            or type(minimum_support) is not int or minimum_support < 4):
        raise ValueError("Invalid selection thresholds")
    candidates = list(candidate_bases)
    if not 1 <= len(candidates) <= 64:
        raise ValueError("Supply between one and 64 candidate reciprocal lattices")
    bases, labels, indices, deviations, likelihood = [], [], [], [], [np.zeros(len(q))]
    for label, candidate in enumerate(candidates):
        basis, _ = reduced_basis(np.asarray(candidate, float))

        def equivalent(existing):
            transform = np.linalg.solve(existing, basis)
            integer = np.rint(transform)
            return (np.max(np.abs(transform-integer)) < 1e-7
                    and abs(abs(np.linalg.det(integer))-1.) < 1e-8)

        if any(equivalent(other) for other in bases):
            continue
        fractional = np.linalg.solve(basis, q.T).T
        nearest = np.rint(fractional)
        if not np.isfinite(nearest).all() or np.any(np.abs(nearest) > 2**52):
            raise ValueError("Candidate indices exceed supported integer range")
        radii = np.maximum(1, np.ceil(.5 + 7.*sigma*np.linalg.norm(np.linalg.inv(basis), axis=1)).astype(int))
        score, distance, offset, _ = periodic_gaussian(
            basis, fractional-nearest, sigma, radii, 2048, 7., max_radius=16)
        bases.append(basis)
        labels.append(label)
        indices.append((nearest + offset).astype(np.int64))
        deviations.append(distance)
        likelihood.append(score)

    ratios = np.column_stack(likelihood)
    chosen, trace = [0], []
    best_score, weights = 0., np.ones(1)
    for _ in range(min(maximum_lattices, len(labels))):
        offers = []
        for column in range(1, ratios.shape[1]):
            if column in chosen:
                continue
            actual = ratios[:, [*chosen, column]]
            trial_weights = _fit_weights(actual[~mask])
            values = actual[mask] + np.log(np.maximum(trial_weights, 1e-15))
            normalizer = logsumexp(values, axis=1)
            posterior = np.exp(values[:, -1] - normalizer)
            support = ((posterior >= .5) & (deviations[column-1][mask] <= 3.*sigma)
                       & np.any(indices[column-1][mask] != 0, axis=1))
            unique = np.unique(indices[column-1][mask][support], axis=0)
            score = float(normalizer.sum())
            if len(unique) >= minimum_support and affine_integer_rank(unique) == 3:
                offers.append((score, column, trial_weights, len(unique)))
        if not offers:
            break
        score, column, trial_weights, support = max(offers, key=lambda entry: entry[0])
        if score-best_score < minimum_gain:
            break
        trace.append({"candidate": labels[column-1], "selection_log_gain": score-best_score,
                      "unique_supported_hkl": support})
        chosen.append(column)
        best_score, weights = score, trial_weights
    return {"selected": [labels[column-1] for column in chosen[1:]], "membership_weights": weights.tolist(),
            "selection_log_ratio": best_score, "trace": trace, "unique_candidates": len(labels),
            "weight_semantics": "Work centroid mixture weights with the outlier component first"}
