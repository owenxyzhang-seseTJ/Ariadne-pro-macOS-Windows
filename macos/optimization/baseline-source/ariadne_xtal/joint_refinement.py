"""Refine a structure directly against compressed overlapping pixel evidence.

For group g, the data term is ||R_g diag(c_g alpha[d_g]) I_g(theta)-z_g||².
R and z retain the Gaussian pixel likelihood after linear background
elimination. No individual component intensity or diagonalized uncertainty is
invented. Domain indices and calibration coefficients must come from the
indexing/integration model. All domains here are the same structural phase.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import least_squares
from scipy.special import softmax

from .derivatives import DiffractionPredictor, right_singular_system
from .model import Structure
from .refinement import (RefinementOptions, _Parameterization, _validate_constraints,
                         _reflection_keys, _json_value)
from .restraints import normalize_restraints, restraint_residuals
from .scattering import _indices, calculated_intensities


MAX_DOMAINS = 256


@dataclass
class GaussianStructureGroup:
    hkl: np.ndarray
    R: np.ndarray
    z: np.ndarray
    residual_constant: float = 0.
    coefficients: np.ndarray | None = None
    domain_ids: np.ndarray | None = None
    role: str = "work"
    label: str = ""
    valid_pixels: int | None = None
    background_rank: int = 0

    def __post_init__(self):
        h = np.asarray(self.hkl, dtype=float)
        if (h.ndim != 2 or h.shape[1] != 3 or not len(h)
                or not np.isfinite(h).all() or not np.equal(h, np.rint(h)).all()):
            raise ValueError("Group hkl must be a nonempty integral N by 3 array")
        self.hkl, _ = _indices(h)
        self.R, self.z = np.asarray(self.R, float), np.asarray(self.z, float)
        # JSON has no way to retain the second dimension of a 0 by K matrix.
        # A zero-information group still carries its profiled residual constant.
        if self.R.shape == (0,):
            self.R = self.R.reshape(0, len(h))
        if (self.R.ndim != 2 or self.R.shape[1] != len(h) or self.z.shape != (len(self.R),)
                or not np.isfinite(self.R).all() or not np.isfinite(self.z).all()):
            raise ValueError("Compressed factor R and target z have inconsistent shapes or values")
        if not np.isfinite(self.residual_constant) or self.residual_constant < 0:
            raise ValueError("Residual constant must be finite and nonnegative")
        if self.role not in {"work", "selection", "audit"}:
            raise ValueError("Group role must be work, selection or audit")
        self.coefficients = np.ones(len(h)) if self.coefficients is None else np.asarray(self.coefficients, float)
        if self.coefficients.shape != (len(h),) or not np.isfinite(self.coefficients).all() or np.any(self.coefficients < 0):
            raise ValueError("Component calibration coefficients must be finite and nonnegative")
        domains = np.zeros(len(h)) if self.domain_ids is None else np.asarray(self.domain_ids, float)
        if (domains.shape != (len(h),) or not np.isfinite(domains).all()
                or np.any(domains < 0) or np.any(domains >= MAX_DOMAINS)
                or not np.equal(domains, np.rint(domains)).all()):
            raise ValueError(f"domain_ids must be integer indices in 0..{MAX_DOMAINS-1}")
        self.domain_ids = domains.astype(int)
        if self.valid_pixels is not None and (isinstance(self.valid_pixels, bool)
                or not isinstance(self.valid_pixels, Integral) or self.valid_pixels < len(self.R)):
            raise ValueError("valid_pixels must cover the compressed data rows")
        if (isinstance(self.background_rank, bool) or not isinstance(self.background_rank, Integral)
                or self.background_rank < 0):
            raise ValueError("background_rank must be a nonnegative integer")
        if self.valid_pixels is not None and self.background_rank + len(self.R) > self.valid_pixels:
            raise ValueError("Signal modes plus background rank exceed the number of valid pixels")

    @classmethod
    def from_observation(cls, observation, hkl, **kwargs):
        """Bridge from ariadne.reduction_native.GaussianObservation."""
        return cls(hkl, observation.R, observation.z, observation.residual_constant,
                   valid_pixels=observation.valid_pixels,
                   background_rank=observation.background_rank, **kwargs)

    def to_dict(self):
        return _json_value({"hkl": self.hkl, "R": self.R, "z": self.z,
                           "residual_constant": self.residual_constant,
                           "coefficients": self.coefficients, "domain_ids": self.domain_ids,
                           "role": self.role, "label": self.label,
                           "valid_pixels": self.valid_pixels, "background_rank": self.background_rank})


def refine_observations(structure: Structure, groups, options=None, *,
                         domain_fractions=None, refine_domain_fractions=False,
                         phase_relation="single_phase", restraints=None, constraints=None):
    """Fit same-phase nonmerohedral or coincident components without detwinning.

    The global domain fractions form a simplex. They can be jointly estimated
    when the complete experiment supplies information; exact overlap alone
    cannot create it. Pixel/profile geometry and calibration are conditioned
    on. Nonlinear or signal-dependent Gaussian variances are not compressed by
    this interface. The separate count likelihood keeps raw photon counts.
    """
    groups = [GaussianStructureGroup(**g) if isinstance(g, dict) else g for g in groups]
    if not groups or any(not isinstance(g, GaussianStructureGroup) for g in groups):
        raise ValueError("Provide GaussianStructureGroup observations")
    work = [g for g in groups if g.role == "work"]
    if not work or not any(len(g.z) for g in work):
        raise ValueError("At least one observable work-group mode is required")
    options = RefinementOptions() if options is None else (
        RefinementOptions(**options) if isinstance(options, dict) else copy.deepcopy(options))
    options.validate()
    if options.twin_laws or options.refine_twin_fractions or options.twin_fractions is not None:
        raise ValueError("Explicit component domains replace merohedral twin_laws in this interface")
    if options.weight_a or options.weight_b or options.covariance_scale != "absolute":
        raise ValueError("Compressed pixel likelihood uses its supplied fixed variances and absolute covariance")
    ndomain = 1 + max(int(g.domain_ids.max()) for g in groups)
    fractions = np.full(ndomain, 1/ndomain) if domain_fractions is None else np.asarray(domain_fractions, float)
    if (fractions.shape != (ndomain,) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("domain_fractions must be a nonnegative simplex including every domain")
    if ndomain > 1 and phase_relation != "same_phase_supported":
        raise ValueError("Multiple domains require phase_relation='same_phase_supported'")
    if phase_relation not in {"single_phase", "same_phase_supported"}:
        raise ValueError("phase_relation must explicitly describe a supported common structural phase")
    if type(refine_domain_fractions) is not bool:
        raise ValueError("refine_domain_fractions must be boolean")
    if refine_domain_fractions and np.any(fractions <= 0):
        raise ValueError("Free domain fractions require strictly positive initial values")
    assigned = {}
    for group in groups:
        for key in _reflection_keys(structure, group.hkl, []):
            if key in assigned and assigned[key] != group.role:
                raise ValueError("Equivalent reflection evidence crosses work/selection/audit roles")
            assigned[key] = group.role
    constraints = _validate_constraints(structure, constraints)
    restraints = normalize_restraints(restraints)
    hkl = np.concatenate([g.hkl for g in work])
    slices, cursor = [], 0
    for g in work:
        slices.append(slice(cursor, cursor+len(g.hkl)))
        cursor += len(g.hkl)
    base = calculated_intensities(structure, hkl)
    forward = np.concatenate([g.R @ (g.coefficients*fractions[g.domain_ids]*base[s])
                              for g, s in zip(work, slices)])
    target = np.concatenate([g.z for g in work])
    if options.scale is None:
        denominator = float(forward @ forward)
        if denominator <= 0:
            raise ValueError("Initial structural model has no observable work intensity; provide an explicit scale or a different model")
        scale = max(1e-12, float(forward @ target / denominator))
    else:
        scale = options.scale
    parameterization = _Parameterization(structure, options, scale, np.ones(1), constraints)
    nbase = len(parameterization.start)
    names = parameterization.names.copy()
    values = np.asarray(parameterization.start)
    lower, upper = np.asarray(parameterization.lower), np.asarray(parameterization.upper)
    free_domains = refine_domain_fractions and ndomain > 1
    if free_domains:
        values = np.r_[values, np.log(fractions[1:]/fractions[0])]
        lower, upper = np.r_[lower, np.full(ndomain-1, -30.)], np.r_[upper, np.full(ndomain-1, 30.)]
        names += [f"domain_logit[{i}]" for i in range(1, ndomain)]
    predictor = DiffractionPredictor(parameterization, hkl, block_size=options.reflection_block_size)
    evaluations = 0

    def evaluate(x, with_jacobian=True):
        nonlocal evaluations
        evaluations += 1
        model, prediction, derivative = predictor.evaluate(x[:nbase], jacobian=with_jacobian)
        alpha = softmax(np.r_[0., x[nbase:]]) if free_domains else fractions
        dalpha = (np.diag(alpha)-np.outer(alpha, alpha))[:, 1:]
        residuals, jacobians = [], []
        for group, section in zip(work, slices):
            multiplier = group.coefficients * alpha[group.domain_ids]
            residuals.append(group.R @ (multiplier*prediction[section]) - group.z)
            if with_jacobian:
                jac = group.R @ (multiplier[:, None]*derivative[section])
                if free_domains:
                    other = group.R @ ((group.coefficients*prediction[section])[:, None]*dalpha[group.domain_ids])
                    jac = np.column_stack([jac, other])
                jacobians.append(jac)
        restraint_vector = restraint_residuals(model, restraints)
        residuals.append(restraint_vector)
        if with_jacobian and len(restraint_vector):
            # Diffraction derivatives are exact. Geometric restraints retain
            # independent central differences in the constrained coordinates.
            jr = np.zeros((len(restraint_vector), len(x)))
            for j in range(nbase):
                step = np.cbrt(np.finfo(float).eps)*max(1., abs(x[j]))
                plus, minus = x[:nbase].copy(), x[:nbase].copy()
                plus[j] += step
                minus[j] -= step
                jr[:, j] = (restraint_residuals(parameterization.unpack(plus)[0], restraints)
                            - restraint_residuals(parameterization.unpack(minus)[0], restraints))/(2*step)
            jacobians.append(jr)
        return np.concatenate(residuals), (np.vstack(jacobians) if with_jacobian else None)

    initial = evaluate(values, False)[0]
    if len(values):
        analytic = options.jacobian == "analytic"
        fit = least_squares(lambda x: evaluate(x, analytic)[0], values,
                            jac=(lambda x: evaluate(x, True)[1]) if analytic else "3-point", bounds=(lower, upper),
                            method="trf", x_scale="jac", max_nfev=options.max_nfev,
                            ftol=options.ftol, xtol=options.xtol, gtol=options.gtol)
        values, success, message, active = fit.x, bool(fit.success), fit.message, fit.active_mask
    else:
        success, message, active = True, "All parameters fixed", np.empty(0, int)
    residual, jacobian = evaluate(values)
    model, scale, _ = parameterization.unpack(values[:nbase])
    predictor.validate_site_multiplicities(model)
    alpha = softmax(np.r_[0., values[nbase:]]) if free_domains else fractions
    singular, vt = right_singular_system(jacobian)
    rank = int(np.count_nonzero(singular > options.covariance_rcond*singular[0])) if len(singular) else 0
    estimable = vt[:rank].T
    pseudo = (estimable/singular[:rank]**2) @ estimable.T if rank else np.zeros((len(values), len(values)))
    identifiable = rank == len(values)
    covariance = pseudo if identifiable else None
    fraction_estimates, fraction_uncertainties, fraction_status = [], [], []
    fraction_gradient = np.zeros((ndomain, len(values)))
    if free_domains:
        fraction_gradient[:, nbase:] = (np.diag(alpha)-np.outer(alpha, alpha))[:, 1:]
    for fraction, gradient in zip(alpha, fraction_gradient):
        unidentifiable = (free_domains and np.linalg.norm(vt[rank:] @ gradient)
                          > 1e-8*np.linalg.norm(gradient))
        on_boundary = np.any(np.abs(gradient[active != 0]) > 0)
        fraction_estimates.append(None if unidentifiable else float(fraction))
        if not free_domains:
            fraction_status.append("fixed")
            fraction_uncertainties.append(None)
        elif unidentifiable:
            fraction_status.append("unidentifiable")
            fraction_uncertainties.append(None)
        elif on_boundary:
            fraction_status.append("boundary")
            fraction_uncertainties.append(None)
        else:
            fraction_status.append("estimable")
            fraction_uncertainties.append(float(np.sqrt(max(gradient @ pseudo @ gradient, 0.))))
    work_rows = sum(len(g.z) for g in work)
    work_singular, _ = right_singular_system(jacobian[:work_rows])
    work_rank = (int(np.count_nonzero(work_singular > options.covariance_rcond*work_singular[0]))
                 if len(work_singular) else 0)
    scores = {role: {"chi_squared": 0., "compressed_modes": 0, "groups": 0} for role in ("work", "selection", "audit")}
    group_diagnostics = []
    for group in groups:
        intensity = calculated_intensities(model, group.hkl, scale=scale)
        component_prediction = group.coefficients*alpha[group.domain_ids]*intensity
        difference = group.R @ component_prediction - group.z
        chi2 = float(difference @ difference + group.residual_constant)
        scores[group.role]["chi_squared"] += chi2
        scores[group.role]["compressed_modes"] += len(group.z)
        scores[group.role]["groups"] += 1
        group_diagnostics.append({"label": group.label, "role": group.role,
                                  "n_components": len(group.hkl), "n_modes": len(group.z),
                                  "chi_squared": chi2, "predicted_components": component_prediction.tolist(),
                                  "prediction_semantics": "structure predictions, not independently measured intensities"})
    constant = sum(g.residual_constant for g in work)
    warnings = parameterization.warnings.copy()
    if not identifiable:
        warnings.append("The structure/domain parameter Jacobian is rank deficient; covariance is undefined")
    if work_rank < rank:
        warnings.append("Some parameter directions are identified by restraints rather than diffraction")
    if np.any(active):
        warnings.append("Some parameters lie on a bound; Gaussian covariance is only an interior curvature diagnostic")
    return _json_value({"schema_version": "ariadne.joint-structure/1", "structure": model.to_dict(),
                        "success": success, "message": message, "scale": scale,
                        "domain_fractions": alpha, "parameter_names": names,
                        "domain_fractions_semantics": "Numerical representative; consult domain_fraction_estimates and domain_fraction_status for inference",
                        "domain_fraction_estimates": fraction_estimates,
                        "domain_fraction_standard_uncertainties": fraction_uncertainties,
                        "domain_fraction_status": fraction_status,
                        "parameters": values, "active_mask": active,
                        "initial_objective": float(initial @ initial + constant),
                        "objective": float(residual @ residual + constant),
                        "scores": scores, "groups": group_diagnostics,
                        "jacobian_rank": rank, "n_parameters": len(values),
                        "diffraction_jacobian_rank": work_rank,
                        "identifiable": identifiable, "singular_values": singular,
                        "nullspace": vt[rank:], "covariance": covariance,
                        "normal_matrix_pseudoinverse": pseudo, "warnings": warnings,
                        "evaluation_count": evaluations,
                        "diffraction_evaluations": predictor.evaluations,
                        "diffraction_derivative_method": options.jacobian,
                        "numeric_backend": predictor.numeric_backend,
                        "numeric_backend_requested": predictor.numeric_backend_requested,
                        "uncertainty_conditioning": "Fixed geometry, profiles, calibration, Gaussian variances and background basis",
                        "component_intensities_were_detwinned": False})
