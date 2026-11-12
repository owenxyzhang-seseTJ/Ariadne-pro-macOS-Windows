"""Observable pixel likelihoods with an explicitly uncertain background.

This module operates on calibrated pixels and supplied profiles, without a
DIALS import.  It does not infer detector calibration or profile geometry.

Gaussian data are compressed by streamed square-root elimination.  For every
physical intensity vector I, ``chi2(I)`` equals the original weighted pixel
residual minimized over the *unconstrained linear* background coefficients.
No positive-intensity clipping, detwinning, or normal-equation solve occurs.

Genuine photon counts use their original Poisson likelihood.  Signal and
nonnegative-background coefficients are fitted jointly; interval evaluation
refits the background.  A general Poisson image has no fixed-size Gaussian
sufficient statistic.  Its pixels are therefore retained and evaluated in
bounded row blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from scipy.linalg import qr
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, xlogy
from scipy.stats import chi2


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _inputs(data, profiles, background, valid):
    y = np.asarray(data, dtype=float)
    p = np.asarray(profiles, dtype=float)
    if y.ndim != 1 or p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] < 1:
        raise ValueError("data must be N values and profiles an N by K matrix, K >= 1")
    b = np.empty((len(y), 0)) if background is None else np.asarray(background, dtype=float)
    if b.ndim != 2 or b.shape[0] != len(y):
        raise ValueError("background must be an N by B matrix")
    if valid is None:
        mask = np.ones(len(y), dtype=bool)
    else:
        mask = np.asarray(valid)
        if mask.dtype != np.bool_ or mask.shape != y.shape:
            raise ValueError("valid must be a boolean pixel mask")
    if not np.any(mask):
        raise ValueError("at least one valid pixel is required")
    # Input arrays remain caller-owned: the only design copies are row blocks.
    return y, p, b, mask


def _rows(mask, chunk_rows):
    for start in range(0, len(mask), chunk_rows):
        stop = min(len(mask), start + chunk_rows)
        selected = mask[start:stop]
        if selected.any():
            # A slice avoids a needless copy for a wholly valid block.
            yield slice(start, stop), slice(None) if selected.all() else selected


def _safe_norm_columns(matrix):
    if not matrix.shape[1]:
        return np.empty(0)
    peak = np.max(np.abs(matrix), axis=0, initial=0.)
    safe = np.where(peak > 0, peak, 1.)
    norm = safe * np.linalg.norm(matrix / safe, axis=0)
    return np.where(norm > 0, norm, 1.)


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


@dataclass
class GaussianObservation:
    """Small likelihood in the original ordered, full-profile intensity units.

    ``R`` is rank by K, not a Cholesky factor of a normal matrix.
    ``nullspace`` contains orthonormal rows in physical intensity coordinates.
    A rank-deficient observation still scores a structural prediction exactly;
    its individual latent intensities need not be measured separately.
    """

    R: np.ndarray
    z: np.ndarray
    residual_constant: float
    rank: int
    nullspace: np.ndarray
    valid_pixels: int
    background_rank: int
    background_count: int
    normalized_singular_values: np.ndarray
    rank_cutoff: float
    beta_offset: np.ndarray
    beta_coupling: np.ndarray
    beta_conditional_covariance: np.ndarray
    _inverse: np.ndarray = field(repr=False)
    _covariance: np.ndarray = field(repr=False)

    @property
    def n_components(self) -> int:
        return self.R.shape[1]

    def _intensities(self, intensities):
        x = np.asarray(intensities, dtype=float)
        if x.shape != (self.n_components,) or not np.isfinite(x).all():
            raise ValueError("intensities must be one finite value per profile")
        return x

    def residual(self, intensities) -> np.ndarray:
        return self.R @ self._intensities(intensities) - self.z

    def chi2(self, intensities) -> float:
        residual = self.residual(intensities)
        return float(residual @ residual + self.residual_constant)

    def gradient(self, intensities) -> np.ndarray:
        return 2 * self.R.T @ self.residual(intensities)

    def estimable(self, contrast, tolerance=1e-9) -> bool:
        contrast = self._intensities(contrast)
        norm = np.linalg.norm(contrast)
        return norm == 0 or np.linalg.norm(self.nullspace @ contrast) <= tolerance * norm

    def contrast(self, contrast) -> dict:
        contrast = self._intensities(contrast)
        identifiable = self.estimable(contrast)
        estimate = self._inverse @ self.z
        return _json({
            "estimable": identifiable,
            "estimate": float(contrast @ estimate) if identifiable else None,
            "variance": float(contrast @ self._covariance @ contrast) if identifiable else None,
            "scope": "fixed calibrated variance and profiles; linear background uncertainty included",
        })

    def fit(self) -> dict:
        """Signed Gaussian fit; unidentifiable components are not published."""
        estimate = self._inverse @ self.z
        identifiable = np.array([self.estimable(row) for row in np.eye(self.n_components)])
        beta = self.beta_offset - self.beta_coupling @ estimate
        information = np.sum(self.R * self.R, axis=0)
        ratio = np.zeros(self.n_components)
        selected = identifiable & (np.diag(self._covariance) > 0) & (information > 0)
        ratio[selected] = 1 / (np.diag(self._covariance)[selected] * information[selected])
        joint_covariance = None
        if self.rank == self.n_components and self.background_rank == self.background_count:
            cross = -self._covariance @ self.beta_coupling.T
            beta_cov = (self.beta_conditional_covariance
                        + self.beta_coupling @ self._covariance @ self.beta_coupling.T)
            joint_covariance = np.block([[self._covariance, cross], [cross.T, beta_cov]])
        return _json({
            "schema": "ariadne-gaussian-pixel-fit/2",
            "intensities": [float(x) if ok else None for x, ok in zip(estimate, identifiable)],
            "intensities_internal": estimate,
            "internal_solution": "minimum Euclidean norm representative; not a detwinning measurement",
            "individual_identifiable": identifiable,
            "background_coefficients_internal": beta,
            "chi2": self.chi2(estimate),
            "covariance": self._covariance if self.rank == self.n_components else None,
            "covariance_identifiable_subspace": self._covariance,
            "joint_covariance_intensity_then_background": joint_covariance,
            "conditional_information_ratio": np.clip(ratio, 0, 1),
            "rank": self.rank,
            "nullspace": self.nullspace,
            "selection": "all valid pixels; no intensity or sign threshold",
        })

    def to_dict(self) -> dict:
        return _json({
            "schema": "ariadne-gaussian-observation/2",
            "R": self.R, "z": self.z, "residual_constant": self.residual_constant,
            "rank": self.rank, "nullspace": self.nullspace,
            "valid_pixels": self.valid_pixels,
            "background_rank": self.background_rank,
            "background_count": self.background_count,
            "normalized_singular_values": self.normalized_singular_values,
            "rank_cutoff": self.rank_cutoff,
            "identity": "min_beta ||W^0.5 (y - P I - B beta)||^2 = ||R I - z||^2 + residual_constant",
            "scope": "fixed variance, mask, geometry and profiles; free linear background profiled",
            "intensity_units": "original full-profile amplitudes; truncated profiles are not renormalized",
            "rank_policy": "floating-point rank of equilibrated design, no scientific intensity cutoff",
        })


def compress_gaussian(data, profiles, *, variance, background=None, valid=None,
                      chunk_rows=4096) -> GaussianObservation:
    """Eliminate background and compress a Gaussian region without an N by N Q.

    Additional design workspace is O(chunk_rows * (K+B+1) + (K+B+1)^2).
    Input arrays are caller-owned; arbitrary masked values are never evaluated.
    The variance is fixed, supplied calibration, not updated from fitted I.
    """
    chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
    y, p, b, mask = _inputs(data, profiles, background, valid)
    v = np.asarray(variance, dtype=float)
    if v.ndim == 0:
        v = np.broadcast_to(v, y.shape)
    if v.shape != y.shape:
        raise ValueError("variance must be scalar or one value per pixel")
    k, nb = p.shape[1], b.shape[1]
    columns = nb + k + 1
    factor = np.empty((0, columns))
    count = 0
    for section, selected in _rows(mask, chunk_rows):
        yy, pp, bb, vv = y[section][selected], p[section][selected], b[section][selected], v[section][selected]
        if (not np.isfinite(yy).all() or not np.isfinite(pp).all() or not np.isfinite(bb).all()
                or not np.isfinite(vv).all() or np.any(vv <= 0)):
            raise ValueError("valid Gaussian rows require finite data/design and positive finite variance")
        weight = 1 / np.sqrt(vv)
        work = np.empty((len(factor) + len(yy), columns), order="F")
        work[:len(factor)] = factor
        rows = work[len(factor):]
        rows[:, :nb] = bb * weight[:, None]
        rows[:, nb:nb + k] = pp * weight[:, None]
        rows[:, -1] = yy * weight
        if not np.isfinite(rows).all():
            raise ValueError("whitened Gaussian rows exceed floating-point range")
        triangular = qr(work, mode="r", overwrite_a=True, check_finite=False)[0]
        factor = triangular[:min(columns, len(triangular))].copy()
        count += len(yy)
    bg, signal, target = factor[:, :nb], factor[:, nb:nb + k], factor[:, -1]
    background_rank = 0
    inverse_background = np.empty((0, len(factor)))
    if nb:
        scale_b = _safe_norm_columns(bg)
        ub, sb, vtb = np.linalg.svd(bg / scale_b, full_matrices=False)
        cutoff_b = np.finfo(float).eps * max(count, nb) * (sb[0] if len(sb) else 0.)
        background_rank = int(np.count_nonzero(sb > cutoff_b))
        basis = ub[:, :background_rank]
        inverse_background = ((vtb[:background_rank].T / sb[:background_rank])
                              @ basis.T) / scale_b[:, None]
        signal = signal - basis @ (basis.T @ signal)
        target = target - basis @ (basis.T @ target)
    beta_offset = inverse_background @ factor[:, -1]
    beta_coupling = inverse_background @ factor[:, nb:nb + k]
    beta_conditional_covariance = inverse_background @ inverse_background.T
    # Tiny residuals of a column wholly in the background subspace are roundoff,
    # not a new observable direction.  Compare before rescaling that residual.
    original_scale = _safe_norm_columns(factor[:, nb:nb + k])
    near_zero = np.linalg.norm(signal, axis=0) <= (8 * np.finfo(float).eps
                                                * max(columns, 1) * original_scale)
    signal[:, near_zero] = 0.
    scales = _safe_norm_columns(signal)
    u, singular, vt = np.linalg.svd(signal / scales, full_matrices=True)
    cutoff = np.finfo(float).eps * max(count, k) * (singular[0] if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > cutoff))
    z = u[:, :rank].T @ target
    r = singular[:rank, None] * vt[:rank] * scales[None, :]
    orthogonal = target - u[:, :rank] @ z
    constant = float(orthogonal @ orthogonal)
    # This second SVD is only rank by K.  It makes physical null vectors and
    # the diagnostic minimum-norm solution independent of column equilibration.
    ur, sr, vtr = np.linalg.svd(r, full_matrices=True)
    inverse = (vtr[:rank].T / sr) @ ur[:, :rank].T if rank else np.zeros((k, 0))
    covariance = inverse @ inverse.T
    return GaussianObservation(r, z, constant, rank, vtr[rank:], count,
                               background_rank, nb, singular, float(cutoff),
                               beta_offset, beta_coupling, beta_conditional_covariance,
                               inverse, covariance)


def positive_background_basis(coordinates) -> np.ndarray:
    """Multilinear Bernstein basis on supplied coordinates, scaled to [0, 1].

    The 2**d corner coefficients are nonnegative expected backgrounds.  This
    represents a constant or gently varying positive background without an
    exponential-link nonlinearity.  Coordinates must be N by d, 1 <= d <= 3.
    """
    x = np.asarray(coordinates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not 1 <= x.shape[1] <= 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("coordinates must be a finite nonempty N by d array, 1 <= d <= 3")
    span = np.ptp(x, axis=0)
    if np.any(span <= 0):
        raise ValueError("each background coordinate must span a nonzero range")
    unit = (x - x.min(axis=0)) / span
    answer = np.ones((len(x), 1))
    for column in unit.T:
        answer = np.concatenate((answer * (1-column[:, None]), answer * column[:, None]), axis=1)
    return answer


def _half_poisson_deviance(counts, mean):
    """Sum mu - n + n log(n/mu), with 0 log 0 = 0 and exact support."""
    if np.any(mean < 0) or not np.isfinite(mean).all():
        return np.inf
    positive = counts > 0
    if np.any(mean[positive] == 0):
        return np.inf
    value = float(np.sum(mean[~positive]))
    if positive.any():
        n, mu = counts[positive], mean[positive]
        delta = (mu-n) / n
        close = np.abs(delta) < .125
        terms = np.empty_like(n)
        # d-log(1+d) = d^2(1/2-d/3+d^2/4-...).
        d = delta[close]
        polynomial = np.full_like(d, 1 / 24)
        for power in range(23, 1, -1):
            polynomial = (-1.)**power / power + d * polynomial
        terms[close] = n[close] * d*d * polynomial
        terms[~close] = mu[~close] - n[~close] + n[~close] * (np.log(n[~close]) - np.log(mu[~close]))
        value += float(np.sum(terms))
    return value


@dataclass
class PoissonFit:
    observation: Any = field(repr=False)
    intensities_internal: np.ndarray
    background_coefficients: np.ndarray
    half_deviance: float
    converged: bool
    kkt_error: float
    message: str
    evaluations: int
    geometry: GaussianObservation = field(repr=False)
    fisher_covariance: np.ndarray | None = field(default=None, repr=False)

    def profile_interval(self, contrast, confidence_level=.95):
        return self.observation.profile_interval(self, contrast, confidence_level)

    def to_dict(self):
        flags = [self.geometry.estimable(row) for row in np.eye(len(self.intensities_internal))]
        return _json({
            "schema": "ariadne-joint-poisson-fit/1",
            "intensities": [float(x) if ok else None for x, ok in zip(self.intensities_internal, flags)],
            "intensities_internal": self.intensities_internal,
            "individual_identifiable": flags,
            "background_coefficients": self.background_coefficients,
            "poisson_deviance": 2*self.half_deviance,
            "converged": self.converged, "kkt_error": self.kkt_error,
            "kkt_coordinates": "initial expected-Fisher diagonal units of integrated-count parameters",
            "message": self.message, "objective_evaluations": self.evaluations,
            "rank_after_background": self.geometry.rank,
            "nullspace": self.geometry.nullspace,
            "fisher_joint_covariance_intensity_then_background": self.fisher_covariance,
            "scope": "genuine independent photon counts; jointly fitted nonnegative linear background",
            "uncertainty": "profile likelihood refits all free signal and background coefficients; LR cutoff is asymptotic",
            "intensity_units": "full-profile expected photon counts; no profile renormalization",
        })


class PoissonObservation:
    """Exact affine Poisson model mu = offset + P I + B beta.

    P and B must be nonnegative, and I,beta are nonnegative.  A supplied offset
    is a *known mean*, not an observed background-subtraction residual.
    Arrays are retained by reference for bounded memory; callers must not
    mutate them while the observation is in use.
    """

    def __init__(self, data, profiles, *, background=None, offset=None, valid=None,
                 chunk_rows=4096, max_iterations=2000):
        self.chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
        self.max_iterations = _positive_integer(max_iterations, "max_iterations")
        self.data, self.profiles, self.background, self.valid = _inputs(data, profiles, background, valid)
        self.offset = np.zeros(len(self.data)) if offset is None else np.asarray(offset, dtype=float)
        if self.offset.shape != self.data.shape:
            raise ValueError("offset must be one known expected count per pixel")
        self.k, self.nb = self.profiles.shape[1], self.background.shape[1]
        scales = np.zeros(self.k + self.nb)
        self.count_total = 0.
        self.nll_data_constant = 0.
        for section, selected in _rows(self.valid, self.chunk_rows):
            y, p, b, off = self._block(section, selected)
            if (not np.isfinite(y).all() or np.any(y < 0) or np.any(y != np.rint(y))
                    or np.any(y >= 2**52)):
                raise ValueError("Poisson data must be genuine finite nonnegative integer counts below 2**52")
            if any(not np.isfinite(a).all() or np.any(a < 0) for a in (p, b, off)):
                raise ValueError("Poisson profiles, background basis and known offset must be finite and nonnegative")
            if np.any((y > 0) & (off == 0) & (p.sum(axis=1) + b.sum(axis=1) == 0)):
                raise ValueError("positive counts fall outside the support of every model component")
            scales[:self.k] += p.sum(axis=0)
            scales[self.k:] += b.sum(axis=0)
            self.count_total += float(y.sum())
            self.nll_data_constant += float(np.sum(gammaln(y+1)-xlogy(y, y)+y))
        self.supported_columns = scales > 0
        self.scales = np.where(scales > 0, scales, 1.)
        self.evaluations = 0

    def _block(self, section, selected):
        return (self.data[section][selected], self.profiles[section][selected],
                self.background[section][selected], self.offset[section][selected])

    def _validate_parameters(self, parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.k+self.nb,) or not np.isfinite(parameters).all() or np.any(parameters < 0):
            raise ValueError("one finite nonnegative parameter per signal and background column is required")
        return parameters

    def mean(self, parameters):
        parameters = self._validate_parameters(parameters)
        result = np.full(len(self.data), np.nan)
        for section, selected in _rows(self.valid, self.chunk_rows):
            _, p, b, off = self._block(section, selected)
            result[section][selected] = off + p @ parameters[:self.k] + b @ parameters[self.k:]
        return result

    def value_gradient(self, parameters):
        """Half deviance and its gradient in the original physical units."""
        parameters = self._validate_parameters(parameters)
        value, gradient = 0., np.zeros(self.k+self.nb)
        self.evaluations += 1
        for section, selected in _rows(self.valid, self.chunk_rows):
            y, p, b, off = self._block(section, selected)
            mu = off + p @ parameters[:self.k] + b @ parameters[self.k:]
            contribution = _half_poisson_deviance(y, mu)
            if not np.isfinite(contribution):
                return np.inf, np.full_like(parameters, np.nan)
            score = 1 - np.divide(y, mu, out=np.zeros_like(y), where=mu > 0)
            value += contribution
            gradient[:self.k] += p.T @ score
            gradient[self.k:] += b.T @ score
        return float(value), gradient

    def _scaled_objective(self, x):
        value, grad = self.value_gradient(x/self.scales)
        return value, grad/self.scales

    def nll(self, parameters):
        """Full negative log probability, including log(count factorial)."""
        return self.value_gradient(parameters)[0] + self.nll_data_constant

    def _start(self):
        result = np.zeros(self.k+self.nb)
        active = np.count_nonzero(self.supported_columns)
        result[self.supported_columns] = max(self.count_total/max(active, 1), 1.)
        return result

    def _solver_scales(self, x):
        """Fixed diagonal Fisher units for one optimizer call.

        Integrated-count coordinates alone can be O(10**12), whereas their
        score is O(10**-6). A unit score step then rounds to zero. Whiten the
        parameter units using the *initial* expected Fisher diagonal. This
        changes only numerical coordinates, not the Poisson likelihood.
        Zero-mean, zero-count rows have no finite interior Fisher curvature;
        they retain their exact objective with a unit-scale fallback.
        """
        parameters = x/self.scales
        diagonal = np.zeros_like(x)
        for section, selected in _rows(self.valid, self.chunk_rows):
            _, p, b, off = self._block(section, selected)
            mu = off + p @ parameters[:self.k] + b @ parameters[self.k:]
            inverse = np.divide(1., mu, out=np.zeros_like(mu), where=mu > 0)
            ps = p/self.scales[None, :self.k]
            bs = b/self.scales[None, self.k:]
            diagonal[:self.k] += np.einsum('ij,i,ij->j', ps, inverse, ps)
            diagonal[self.k:] += np.einsum('ij,i,ij->j', bs, inverse, bs)
        scale = np.ones_like(x)
        positive = (diagonal > 0) & np.isfinite(diagonal)
        scale[positive] = 1/np.sqrt(diagonal[positive])
        if not np.isfinite(scale).all():
            raise ValueError("Poisson parameter curvature exceeds floating-point range")
        return scale

    def _minimize(self, fixed=None, start=None):
        """Convex optimization in column-integrated-count coordinates."""
        fixed = {} if fixed is None else dict(fixed)
        base = self._start() if start is None else np.asarray(start, dtype=float).copy()
        for index, value in fixed.items():
            base[index] = value
        free = np.array([j for j in range(len(base)) if j not in fixed and self.supported_columns[j]], dtype=int)
        if not len(free):
            value, _ = self._scaled_objective(base)
            return base, value, bool(np.isfinite(value)), 0., "all available parameters fixed"
        if not np.isfinite(self._scaled_objective(base)[0]):
            base[free] = np.maximum(base[free], 1.)
        if not np.isfinite(self._scaled_objective(base)[0]):
            return base, np.inf, False, np.inf, "fixed parameters remove support for positive counts"

        coordinate_scale = self._solver_scales(base)[free]

        def objective(free_values):
            x = base.copy()
            x[free] = free_values*coordinate_scale
            value, gradient = self._scaled_objective(x)
            return value, gradient[free]*coordinate_scale

        result = minimize(objective, base[free]/coordinate_scale, jac=True, method="L-BFGS-B",
                          bounds=[(0., None)]*len(free),
                          options={"maxiter": self.max_iterations, "maxls": 50,
                                   "ftol": 1e-15, "gtol": 2e-10})
        base[free] = result.x*coordinate_scale
        value, gradient = self._scaled_objective(base)
        # t-max(0,t-g) == min(t,g) for t>=0. Use the latter: the former
        # falsely vanishes when a nonzero g is below the rounding unit of t.
        score = gradient[free]*coordinate_scale
        kkt = float(np.max(np.abs(np.minimum(result.x, score))))
        tolerance = max(2e-7, 8*np.finfo(float).eps*max(float(np.max(result.x)), 1.))
        return base, value, bool(np.isfinite(value) and kkt <= tolerance), kkt, str(result.message)

    def geometry(self):
        """Structural identifiability; no fitted count is used as a weight."""
        return compress_gaussian(np.zeros_like(self.data), self.profiles, variance=1.,
                                 background=self.background, valid=self.valid, chunk_rows=self.chunk_rows)

    def fit(self, initial=None):
        start = None if initial is None else self._validate_parameters(initial)*self.scales
        before = self.evaluations
        x, value, converged, kkt, message = self._minimize(start=start)
        parameters = x/self.scales
        geometry = self.geometry()
        covariance = None
        mean = self.mean(parameters)
        # Interior Fisher curvature is a diagnostic; boundary intervals use the
        # original likelihood and never a fabricated Gaussian standard error.
        if (converged and np.all(parameters > 1e-8) and np.all(mean[self.valid] > 0)
                and geometry.rank == self.k and geometry.background_rank == self.nb):
            fisher = compress_gaussian(np.zeros_like(self.data), self.profiles, variance=mean,
                                       background=self.background, valid=self.valid,
                                       chunk_rows=self.chunk_rows)
            covariance_values = fisher.fit()["joint_covariance_intensity_then_background"]
            covariance = None if covariance_values is None else np.asarray(covariance_values)
        return PoissonFit(self, parameters[:self.k], parameters[self.k:], value, converged,
                          kkt, message, self.evaluations-before, geometry, covariance)

    def profile_background(self, intensities, initial_background=None, *, include_mean=False):
        """Score a structural prediction while refitting only its background.

        ``gradient`` is the envelope gradient with respect to physical I,
        valid where the optimum is differentiable.  A changing active set is
        reported and should not be treated as a globally smooth Hessian.
        This numerical interface returns arrays and preserves infinite losses;
        use a separate JSON serializer only at an artifact boundary.
        """
        intensity = np.asarray(intensities, dtype=float)
        if intensity.shape != (self.k,) or not np.isfinite(intensity).all() or np.any(intensity < 0):
            raise ValueError("fixed physical intensities must be finite and nonnegative")
        start = self._start()
        if initial_background is not None:
            bg = np.asarray(initial_background, dtype=float)
            if bg.shape != (self.nb,) or not np.isfinite(bg).all() or np.any(bg < 0):
                raise ValueError("initial_background has invalid coefficients")
            start[self.k:] = bg*self.scales[self.k:]
        fixed = {j: value*self.scales[j] for j, value in enumerate(intensity)}
        x, value, success, kkt, message = self._minimize(fixed=fixed, start=start)
        parameters = x/self.scales
        _, gradient = self.value_gradient(parameters)
        result = {"half_deviance": value, "nll": value+self.nll_data_constant,
                  "gradient": gradient[:self.k],
                  "background_coefficients": parameters[self.k:],
                  "background_active_zero": parameters[self.k:] <= 1e-8,
                  "converged": success, "kkt_error": kkt, "message": message,
                  "kkt_coordinates": "initial expected-Fisher diagonal units of integrated-count parameters"}
        if include_mean:
            result["mean"] = self.mean(parameters)
        return result

    def _profile_contrast(self, contrast, target, fit):
        coefficient = np.r_[contrast, np.zeros(self.nb)]/self.scales
        selected = np.flatnonzero(coefficient > 0)
        start = np.r_[fit.intensities_internal, fit.background_coefficients]*self.scales
        if target == 0 or len(selected) == 1:
            fixed = ({int(j): 0. for j in selected} if target == 0 else
                     {int(selected[0]): target/coefficient[selected[0]]})
            return self._minimize(fixed=fixed, start=start)[1:3]
        # For an estimable nonnegative sum, retain the exact linear constraint.
        previous = float(coefficient @ start)
        if previous > 0:
            start[selected] *= target/previous
        else:
            start[selected] = target/(len(selected)*coefficient[selected])
        coordinate_scale = self._solver_scales(start)
        transformed = coefficient*coordinate_scale
        norm = float(np.max(transformed))
        row, rhs = transformed/norm, target/norm
        constraint = {"type": "eq", "fun": lambda t: float(row @ t-rhs), "jac": lambda t: row}

        def objective(t):
            value, gradient = self._scaled_objective(t*coordinate_scale)
            return value, gradient*coordinate_scale

        result = minimize(objective, start/coordinate_scale, jac=True, method="SLSQP",
                          bounds=[(0., None)]*len(start), constraints=[constraint],
                          options={"maxiter": self.max_iterations, "ftol": 1e-11})
        feasibility_tolerance = max(1e-8, 16*np.finfo(float).eps*max(abs(rhs), 1.))
        feasible = abs(float(row @ result.x-rhs)) <= feasibility_tolerance
        value = objective(np.maximum(result.x, 0.))[0]
        return value, bool(result.success and feasible and np.isfinite(value))

    def profile_interval(self, fit, contrast, confidence_level=.95):
        """Conditional LR interval for an estimable nonnegative contrast.

        All other intensities AND background coefficients are reoptimized.
        The chi-square(1) cutoff is asymptotic, including at a boundary; this
        function does not claim exact frequentist coverage for sparse counts.
        """
        if fit.observation is not self:
            raise ValueError("fit must originate from this PoissonObservation")
        c = np.asarray(contrast, dtype=float)
        if c.shape != (self.k,) or not np.isfinite(c).all() or np.any(c < 0) or not np.any(c):
            raise ValueError("contrast must be a finite nonnegative nonzero K-vector")
        if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        result = {"contrast": c.tolist(), "confidence_level": float(confidence_level),
                  "threshold": float(chi2.ppf(confidence_level, 1)), "lower": None, "upper": None,
                  "scope": "profiles, calibration and known offset fixed; other intensities and background reoptimized",
                  "calibration": "chi-square(1) asymptotic LR; no exact sparse-count or boundary coverage claim"}
        if not fit.geometry.estimable(c):
            return dict(result, status="not_separately_identifiable", estimate=None)
        if not fit.converged:
            return dict(result, status="fit_not_converged", estimate=None)
        estimate = float(c @ fit.intensities_internal)
        cache = {estimate: 0.}
        failures = []

        def root(target):
            if target not in cache:
                value, success = self._profile_contrast(c, target, fit)
                if not success and np.isfinite(value):
                    failures.append(float(target))
                cache[target] = 2*(value-fit.half_deviance)
            return cache[target]-result["threshold"]

        at_zero = root(0.)
        lower = 0. if at_zero <= 0 or estimate == 0 else brentq(root, 0., estimate, xtol=1e-9)
        upper_test = max(estimate + 1., 2*estimate)
        for _ in range(80):
            if root(upper_test) >= 0:
                break
            upper_test = 2*upper_test+1.
        else:
            return dict(result, estimate=estimate, lower=float(lower), status="upper_not_bracketed")
        upper = brentq(root, estimate, upper_test, xtol=1e-9)
        if failures:
            return dict(result, estimate=estimate, status="profile_fit_not_converged",
                        failed_targets=failures)
        return dict(result, estimate=estimate, lower=float(lower), upper=float(upper),
                    status="ok", profile_evaluations=len(cache))


def fit_poisson(data, profiles, *, background=None, offset=None, valid=None,
                chunk_rows=4096, max_iterations=2000, initial=None) -> PoissonFit:
    """Convenience entry point for the joint count/background model."""
    observation = PoissonObservation(data, profiles, background=background, offset=offset,
                                     valid=valid, chunk_rows=chunk_rows, max_iterations=max_iterations)
    return observation.fit(initial=initial)


def route_region(profiles, *, noise, background=None, variance=None, valid=None,
                 expected_counts=None, chunk_rows=4096) -> dict:
    """Choose a likelihood from calibration and geometry, never measured I/sigma.

    ``noise`` needs a nonempty ``source`` and ``kind`` in {gaussian, poisson,
    poisson_gaussian}.  Gaussian calibration additionally declares
    ``variance_fixed=True``.  Exact Poisson routing requires
    ``poisson_counts_verified=True``, gain=1 and readout_variance=0.  ADU with
    readout noise are directed to the existing calibrated convolution backend.
    Optional expected counts must have independent provenance; they describe
    a computational regime and do not switch off the exact count likelihood.
    """
    if not isinstance(noise, dict) or not isinstance(noise.get("source"), str) or not noise["source"].strip():
        raise ValueError("noise calibration source is required")
    kind = noise.get("kind")
    if kind not in {"gaussian", "poisson", "poisson_gaussian"}:
        raise ValueError("noise kind must be gaussian, poisson or poisson_gaussian")
    p = np.asarray(profiles, dtype=float)
    if p.ndim != 2:
        raise ValueError("profiles must be an N by K array")
    if kind == "gaussian":
        if noise.get("variance_fixed") is not True or variance is None:
            raise ValueError("Gaussian routing requires explicitly fixed calibrated variance")
        weights = variance
        action = "gaussian_observable_likelihood"
    else:
        weights = 1.
        if kind == "poisson":
            if (noise.get("poisson_counts_verified") is not True or noise.get("gain") != 1
                    or noise.get("readout_variance") != 0):
                raise ValueError("Poisson routing requires verified genuine counts, unit count gain and zero readout variance")
            _, p, b, mask = _inputs(np.zeros(len(p)), p, background, valid)
            for section, selected in _rows(mask, _positive_integer(chunk_rows, "chunk_rows")):
                if any(not np.isfinite(a).all() or np.any(a < 0)
                       for a in (p[section][selected], b[section][selected])):
                    raise ValueError("Poisson routing requires nonnegative finite profiles and background basis")
            action = "joint_poisson_background"
        else:
            gain, readout = noise.get("gain"), noise.get("readout_variance")
            if (not isinstance(gain, (int, float)) or not np.isfinite(gain) or gain <= 0
                    or not isinstance(readout, (int, float)) or not np.isfinite(readout) or readout <= 0):
                raise ValueError("Poisson-Gaussian routing requires positive calibrated gain and readout variance")
            action = "existing_calibrated_poisson_gaussian_backend"
    geometry = compress_gaussian(np.zeros(p.shape[0]), p, variance=weights, background=background,
                                 valid=valid, chunk_rows=chunk_rows)
    fitted_geometry = geometry.fit()
    minimum_ratio = min(fitted_geometry["conditional_information_ratio"])
    regime = "unspecified_count_regime"
    if expected_counts is not None:
        if not noise.get("expected_counts_source"):
            raise ValueError("expected counts used for planning require independent provenance")
        mean = np.asarray(expected_counts, dtype=float)
        if mean.shape != (p.shape[0],) or not np.isfinite(mean).all() or np.any(mean < 0):
            raise ValueError("expected_counts must be one finite nonnegative mean per pixel")
        regime = "sparse_counts" if float(np.median(mean)) < 10 else "regular_counts"
    overlap = ("unresolved_combinations" if geometry.rank < p.shape[1] else
               "ill_conditioned_overlap" if minimum_ratio < 1e-3 else "separable_profiles")
    return _json({
        "schema": "ariadne-region-route/1", "likelihood_action": action,
        "overlap_regime": overlap, "count_regime": regime,
        "rank_after_background": geometry.rank, "components": p.shape[1],
        "conditional_information_ratio": fitted_geometry["conditional_information_ratio"],
        "selection": "none; no measured intensity, sign or I/sigma threshold is an input",
        "strong_weak_policy": "use well-resolved references for geometry; retain all regions in their calibrated likelihood",
        "geometry_rank_scope": "supplied profiles and linear-background span",
        "noise_source": noise["source"],
    })
