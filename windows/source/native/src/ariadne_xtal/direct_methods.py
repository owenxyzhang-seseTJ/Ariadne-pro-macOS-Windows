"""Small, explicit operators used by native dual-space phase retrieval.

This module implements mathematical operators, not the SHELXT algorithm.
Triplet phases follow the classic Cochran/Karle--Hauptman approximation.
Their role is to guide a nonconvex search; they are not observations and their
residual is not a probability that the resulting atomic model is correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def intensity_amplitude_prox(amplitude, intensity, sigma, variance=0.25):
    r"""Global radial minimizer for a signed Gaussian intensity observation.

    For ``a=abs(z)`` this returns ``r >= 0`` minimizing

    .. math:: (r-a)^2/(2\tau) + (r^2-y)^2/(2\sigma^2).

    The complex update is ``r * z/abs(z)``.  With ``b=sigma**2/(2*tau)``,
    the positive stationary point solves ``r**3 + (b-y)*r - b*a = 0``.
    There is exactly one positive root when ``a > 0``.  If ``a == 0``,
    compare the stationary points explicitly: the global minimum is
    ``sqrt(max(y-b, 0))``.  The implementation evaluates the positive root
    using scaled hyperbolic/Cardano expressions, avoiding per-reflection
    numerical optimizers, cancellation for large positive ``b-y`` and a
    square of an unscaled large standard uncertainty.

    ``sigma`` is the standard uncertainty of *intensity*, not amplitude.
    Signed ``intensity`` values enter intact.  ``variance`` is an algorithmic
    proximity scale in amplitude-squared units, not an experimental variance.
    """
    a, y, s = np.broadcast_arrays(np.asarray(amplitude, dtype=float),
                                  np.asarray(intensity, dtype=float),
                                  np.asarray(sigma, dtype=float))
    if (not np.isfinite(a).all() or np.any(a < 0) or not np.isfinite(y).all()
            or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError("Amplitude must be nonnegative and sigma positive; all inputs must be finite")
    if not np.isscalar(variance) or not np.isfinite(variance) or variance <= 0:
        raise ValueError("Proximity variance must be a finite positive scalar")
    noise_scale = s / np.sqrt(2 * float(variance))
    scale = np.maximum.reduce([a, np.sqrt(np.abs(y)), noise_scale,
                               np.full(a.shape, np.finfo(float).tiny**0.25)])
    if not np.isfinite(scale).all():
        raise ValueError("Proximity inputs exceed floating-point range")
    aa, yy, b = a / scale, (y / scale) / scale, (noise_scale / scale)**2
    p, v = b - yy, b * aa
    root = np.zeros(a.shape, dtype=float)
    zero = v == 0
    # This also handles representationally vanishing noise at a=0.
    root[zero] = np.sqrt(np.maximum(-p[zero], 0))
    # A cubic-dominated root also avoids sqrt(p/3) underflow when p is a
    # nonzero subnormal.  The omitted linear term is below root roundoff.
    cubic_root = np.cbrt(v)
    flat = (~zero) & (np.abs(p) <= 4 * np.finfo(float).eps * cubic_root**2)
    root[flat] = np.cbrt(v[flat])
    positive = (~zero) & (~flat) & (p > 0)
    if np.any(positive):
        # 2 sqrt(p/3) sinh(asinh(v/(2*(p/3)**1.5))/3).
        # Compute the asinh argument in log space for very small p.
        q = np.sqrt(p[positive] / 3)
        log_arg = np.log(v[positive]) - np.log(2.) - 3 * np.log(q)
        asinh_arg = np.empty_like(log_arg)
        large = log_arg > 20
        asinh_arg[large] = log_arg[large] + np.log(2.)
        asinh_arg[~large] = np.arcsinh(np.exp(log_arg[~large]))
        root[positive] = 2 * q * np.sinh(asinh_arg / 3)
    negative = (~zero) & (~flat) & (p < 0)
    if np.any(negative):
        pp, vv = p[negative], v[negative]
        rr = np.empty_like(pp)
        q = np.sqrt(-pp / 3)
        # Compare the discriminant in logarithmic form. Directly evaluating
        # v**2/4+p**3/27 can underflow both terms to zero and pick the wrong
        # branch even though the correct positive root is representable.
        log_arg = np.log(vv) - np.log(2.) - 3 * np.log(q)
        three = log_arg <= 0
        angle = np.arccos(np.clip(np.exp(log_arg[three]), 0, 1)) / 3
        rr[three] = 2 * q[three] * np.cos(angle)
        single = ~three
        acosh_arg = np.empty(np.count_nonzero(single))
        single_log = log_arg[single]
        large = single_log > 20
        acosh_arg[large] = single_log[large] + np.log(2.)
        acosh_arg[~large] = np.arccosh(np.exp(single_log[~large]))
        rr[single] = 2 * q[single] * np.cosh(acosh_arg / 3)
        root[negative] = rr
    return root * scale


@dataclass(frozen=True)
class TripletGraph:
    """Sparse ``h = k + l`` relations; Hermitian mates are updated together."""

    target: np.ndarray
    left: np.ndarray
    right: np.ndarray
    weight: np.ndarray
    mate: np.ndarray
    size: int
    selected_reflections: int

    def to_dict(self):
        return {"triplet_count": int(len(self.target)),
                "selected_reflections": int(self.selected_reflections),
                "array_bytes": int(sum(a.nbytes for a in
                                      (self.target, self.left, self.right, self.weight, self.mate))),
                "phase_relation": "phi(h) - phi(k) - phi(h-k)",
                "weight_semantics": "heuristic reliability-tempered triplet concentration"}


def build_triplet_graph(hkl, normalized_amplitude, reliability=None, *,
                        maximum_reflections=256, per_reflection=32):
    r"""Select a bounded graph of the strongest available phase relations.

    Relations satisfy exact integer vector addition, so the update is
    invariant to origin shifts.  Both Friedel signs must be present, and no
    index may be repeated.  Only one target of each Friedel pair is stored.
    The weights are ``|E_h E_k E_l| r_h r_k r_l``.  A common concentration
    factor cancels in the tangent direction; this deliberately does not
    interpret unequal-atom approximate triplet weights as exact posteriors.
    ``r_h`` tempers uncertain observations.  Negative measured intensities
    may therefore be absent from this *search guide* while remaining in the
    signed-intensity data likelihood used by the phasing solver.
    """
    raw = np.asarray(hkl)
    amp = np.asarray(normalized_amplitude, dtype=float)
    if (raw.ndim != 2 or raw.shape[1] != 3 or not len(raw)
            or not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all()):
        raise ValueError("Triplet indices must be a nonempty finite integer N by 3 array")
    hk = raw.astype(np.int64)
    if amp.shape != (len(hk),) or not np.isfinite(amp).all() or np.any(amp < 0):
        raise ValueError("Triplet amplitudes must be finite and nonnegative")
    rel = np.ones(len(hk)) if reliability is None else np.asarray(reliability, dtype=float)
    if rel.shape != amp.shape or not np.isfinite(rel).all() or np.any((rel < 0) | (rel > 1)):
        raise ValueError("Reliability must have one value in [0,1] per reflection")
    for name, value, high in (("maximum_reflections", maximum_reflections, 4096),
                              ("per_reflection", per_reflection, 512)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= high:
            raise ValueError(f"{name} must be an integer in 1..{high}")
    lookup = {tuple(h): i for i, h in enumerate(hk)}
    if len(lookup) != len(hk) or (0, 0, 0) in lookup:
        raise ValueError("Triplet indices must be unique and exclude 000")
    try:
        mate = np.array([lookup[tuple(-h)] for h in hk], dtype=np.int32)
    except KeyError as exc:
        raise ValueError("Triplet indices must include both Friedel signs") from exc
    if not np.allclose(amp, amp[mate], rtol=1e-12, atol=1e-12) or not np.allclose(rel, rel[mate]):
        raise ValueError("Triplet amplitudes and reliability must be Friedel symmetric")
    canonical = np.flatnonzero(np.arange(len(hk)) < mate)
    # Deterministic tie handling, preserving complete Friedel pairs.
    strength = amp * rel
    order = np.argsort(-strength[canonical], kind="stable")
    pair_count = min(len(canonical), maximum_reflections // 2)
    chosen = canonical[order[:pair_count]]
    chosen = chosen[strength[chosen] > 0]
    selected = np.sort(np.r_[chosen, mate[chosen]])
    selected_set = set(map(int, selected))
    targets, lefts, rights, weights = [], [], [], []
    for i in chosen:
        candidates = []
        for j in selected:
            k = lookup.get(tuple(hk[i] - hk[j]))
            if k is None or k not in selected_set or j > k:
                continue
            weight = float(strength[i] * strength[j] * strength[k])
            if weight > 0:
                candidates.append((weight, int(j), int(k)))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for weight, j, k in candidates[:per_reflection]:
            targets.append(int(i))
            lefts.append(j)
            rights.append(k)
            weights.append(weight)
    return TripletGraph(np.asarray(targets, dtype=np.int32), np.asarray(lefts, dtype=np.int32),
                        np.asarray(rights, dtype=np.int32), np.asarray(weights, dtype=float),
                        mate, len(hk), len(selected))


def tangent_update(phases, graph: TripletGraph, strength=0.1):
    r"""Damped circular tangent update with automatic Hermitian pairing.

    ``phi_h <- arg((1-lambda) exp(i phi_h) + lambda T_h)`` where
    ``T_h = sum w_hkl exp(i(phi_k+phi_l)) / sum w_hkl``.  Its magnitude
    expresses consistency: contradictory relations cause less movement.
    The full tangent formula is obtained with strength=1.  A weak damped
    update is appropriate as an early search guide; it is annealed to zero
    in ``hybrid`` so final intensities are not forced to satisfy approximate
    triplet identities exactly.
    """
    phi = np.asarray(phases, dtype=float)
    if phi.shape != (graph.size,) or not np.isfinite(phi).all():
        raise ValueError("One finite phase is required per graph reflection")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Tangent strength must lie in [0,1]")
    if not np.allclose(np.exp(1j * phi), np.exp(-1j * phi[graph.mate]), atol=1e-9):
        raise ValueError("Tangent input phases must be Hermitian")
    if not len(graph.target) or strength == 0:
        return phi.copy()
    phase_sum = phi[graph.left] + phi[graph.right]
    real = np.bincount(graph.target, weights=graph.weight * np.cos(phase_sum), minlength=graph.size)
    imag = np.bincount(graph.target, weights=graph.weight * np.sin(phase_sum), minlength=graph.size)
    total = np.bincount(graph.target, weights=graph.weight, minlength=graph.size)
    used = np.flatnonzero(total > 0)
    guidance = (real[used] + 1j * imag[used]) / total[used]
    z = (1 - strength) * np.exp(1j * phi[used]) + strength * guidance
    # If the circular mean is exactly zero, retain the previous phase.
    phi_new = np.where(np.abs(z) > 1e-15, np.angle(z), phi[used])
    result = phi.copy()
    result[used] = phi_new
    result[graph.mate[used]] = -phi_new
    return result


def triplet_residual(phases, graph: TripletGraph):
    """Weighted mean 1-cos(triplet); a search diagnostic, not a likelihood."""
    phi = np.asarray(phases, dtype=float)
    if phi.shape != (graph.size,) or not np.isfinite(phi).all():
        raise ValueError("One finite phase is required per graph reflection")
    if not len(graph.target):
        return None
    difference = phi[graph.target] - phi[graph.left] - phi[graph.right]
    return float(np.sum(graph.weight * (1 - np.cos(difference))) / np.sum(graph.weight))
