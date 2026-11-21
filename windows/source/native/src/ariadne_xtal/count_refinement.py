"""Experimental structure refinement against calibrated Poisson pixels."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .derivatives import DiffractionPredictor
from .pixel_profiles import GaussianPixels
from .refinement import RefinementOptions, _Parameterization, _reflection_keys, _validate_constraints
from .scattering import _indices


@dataclass
class CountStructureGroup:
    hkl: np.ndarray
    observation: object
    coefficients: np.ndarray | None = None
    role: str = "work"
    audit_background: np.ndarray | None = None
    profile_model: GaussianPixels | dict | None = None
    profile_group: str = "shared"

    def __post_init__(self):
        from ariadne.reduction_native import PoissonObservation

        self.hkl, _ = _indices(self.hkl)
        nominal = None
        if isinstance(self.profile_model, dict):
            self.profile_model = GaussianPixels(**self.profile_model)
        if self.profile_model is not None:
            if not isinstance(self.profile_model, GaussianPixels) or not isinstance(self.profile_group, str) or not self.profile_group:
                raise ValueError("provide a Gaussian pixel model and a nonempty profile group")
            nominal = self.profile_model.evaluate(np.zeros(self.profile_model.ndim + 1))
        if isinstance(self.observation, dict):
            observation = dict(self.observation)
            if nominal is not None:
                observation.setdefault("profiles", nominal)
            self.observation = PoissonObservation(**observation)
        if not isinstance(self.observation, PoissonObservation) or len(self.hkl) != self.observation.k:
            raise ValueError("Each count group requires one HKL per calibrated pixel profile")
        if nominal is not None:
            valid = self.observation.valid
            if nominal.shape != self.observation.profiles.shape or not np.allclose(
                    nominal[valid], self.observation.profiles[valid], rtol=1e-10, atol=1e-14):
                raise ValueError("nominal Gaussian fractions must match the supplied signal profiles")
        self.coefficients = (np.ones(len(self.hkl)) if self.coefficients is None
                             else np.asarray(self.coefficients, float))
        if (self.coefficients.shape != (len(self.hkl),) or not np.isfinite(self.coefficients).all()
                or np.any(self.coefficients < 0)):
            raise ValueError("Supply finite nonnegative component scales")
        if self.role not in {"work", "audit"}:
            raise ValueError("Count groups must be work or audit")
        if self.role == "audit":
            self.audit_background = np.asarray(self.audit_background, float)
            if (self.audit_background.shape != (self.observation.nb,)
                    or not np.isfinite(self.audit_background).all() or np.any(self.audit_background < 0)):
                raise ValueError("Audit pixels require independently specified nonnegative background coefficients")


def count_objective(predictor, groups, parameters, *, sections=None):
    """Profile the work backgrounds and differentiate through the structure factor."""
    nbase = len(predictor.parameterization.names)
    _, predictions, jacobian = predictor.evaluate(parameters[:nbase])
    total = 0.
    gradient = np.zeros(len(parameters))
    background_kkt = 0.
    start = 0
    for group in groups:
        stop = start + len(group.hkl)
        intensity = group.coefficients * predictions[start:stop]
        observation = group.observation
        section = sections[group.profile_group] if sections is not None and group.profile_model is not None else None
        if section is not None:
            pixels, pixel_jacobian = group.profile_model.evaluate(parameters[section], derivatives=True)
            observation = observation.with_profiles(pixels)
        profile = observation.profile_background(intensity, include_mean=section is not None)
        if not profile["converged"]:
            raise RuntimeError("Poisson background profile did not converge: " + profile["message"])
        score = profile["gradient"] * group.coefficients
        gradient[:nbase] += jacobian[start:stop].T @ score
        if section is not None:
            mask = observation.valid
            mean = profile["mean"][mask]
            counts = observation.data[mask]
            residual = 1. - np.divide(counts, mean, out=np.zeros_like(mean), where=mean > 0)
            gradient[section] += np.einsum("p,pkd,k->d", residual, pixel_jacobian[mask], intensity)
        total += profile["half_deviance"]
        background_kkt = max(background_kkt, profile["kkt_error"])
        start = stop
    return total, gradient, background_kkt


def _profile_layout(groups, nbase, options):
    if not isinstance(options, dict) or set(options) - {"max_shift", "max_width_ratio"}:
        raise ValueError("profile refinement requires an object with max_shift and max_width_ratio")
    max_shift = options.get("max_shift", 1.)
    ratio = options.get("max_width_ratio", 1.5)
    if (type(max_shift) not in {int, float} or type(ratio) not in {int, float}
            or not np.isfinite([max_shift, ratio]).all() or max_shift <= 0 or ratio <= 1.):
        raise ValueError("profile bounds require a positive shift and a width ratio above one")
    models = {}
    for group in groups:
        if group.role == "work" and group.profile_model is not None:
            key, ndim = group.profile_group, group.profile_model.ndim
            if models.setdefault(key, ndim) != ndim:
                raise ValueError("a shared profile group must use identical axes")
    if not models:
        raise ValueError("profile refinement requires a pixel geometry in the work data")
    for group in groups:
        if group.profile_model is not None and models.get(group.profile_group) != group.profile_model.ndim:
            raise ValueError("every audit profile group must have matching work geometry")
    sections, bounds = {}, []
    for key, ndim in models.items():
        start = nbase + len(bounds)
        bounds.extend([(-max_shift, max_shift)] * ndim)
        bounds.append((-np.log(ratio), np.log(ratio)))
        sections[key] = slice(start, start + ndim + 1)
    return sections, bounds


def refine_count_observations(structure, groups, options=None, constraints=None, *, profile_refinement=None):
    """Fit a candidate structure against calibrated counts.

    The coefficients include fixed relative exposure, scale corrections and
    any supplied domain shares. Work background is reoptimized locally. An
    optional Gaussian model shares shifts and a width ratio within named scans.
    Audit scores use the independently supplied background and work parameters.
    """
    groups = [CountStructureGroup(**g) if isinstance(g, dict) else g for g in groups]
    if any(not isinstance(g, CountStructureGroup) for g in groups):
        raise TypeError("Provide CountStructureGroup observations")
    options = (RefinementOptions(scale=1.) if options is None else
               RefinementOptions(**options) if isinstance(options, dict) else deepcopy(options))
    options.validate()
    if (options.scale is None or options.twin_laws or options.refine_twin_fractions
            or options.twin_fractions is not None or options.weight_a or options.weight_b
            or options.jacobian_layout != "dense" or options.jacobian != "analytic"):
        raise ValueError("Count refinement requires explicit initial scale, fixed component domains and analytic derivatives")
    work = [g for g in groups if g.role == "work"]
    audit = [g for g in groups if g.role == "audit"]
    if not work:
        raise ValueError("At least one work count group is required")
    if {key for g in work for key in _reflection_keys(structure, g.hkl)} & {
            key for g in audit for key in _reflection_keys(structure, g.hkl)}:
        raise ValueError("Equivalent intensity evidence crosses work/audit roles")
    parameterization = _Parameterization(structure, options, options.scale, np.ones(1),
                                          _validate_constraints(structure, constraints))
    predictor = DiffractionPredictor(parameterization, np.concatenate([g.hkl for g in work]),
                                    block_size=options.reflection_block_size)
    values = np.asarray(parameterization.start)
    nbase = len(values)
    bounds = list(zip(parameterization.lower, parameterization.upper))
    sections = None
    if profile_refinement is not None:
        sections, profile_bounds = _profile_layout(groups, nbase, profile_refinement)
        bounds += profile_bounds
        values = np.r_[values, np.zeros(len(profile_bounds))]

    last_values = None
    last_result = None

    def objective(x):
        nonlocal last_values, last_result
        if last_values is None or not np.array_equal(x, last_values):
            last_result = count_objective(predictor, work, x, sections=sections)
            last_values = x.copy()
        return last_result

    initial, _, _ = objective(values)
    if len(values):
        fit = minimize(lambda x: objective(x)[:2], values, jac=True, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxfun": options.max_nfev, "maxiter": options.max_nfev,
                                "ftol": options.ftol, "gtol": options.gtol})
        values, success, message = fit.x, bool(fit.success), str(fit.message)
    else:
        success, message = True, "All structural parameters fixed"
    final, gradient, kkt = objective(values)
    model, scale, _ = parameterization.unpack(values[:nbase])
    predictor.validate_site_multiplicities(model)
    audit_loss = None
    if audit:
        from .scattering import calculated_intensities

        audit_loss = 0.
        for group in audit:
            observation = group.observation
            if sections is not None and group.profile_model is not None:
                observation = observation.with_profiles(group.profile_model.evaluate(values[sections[group.profile_group]]))
            intensity = group.coefficients * calculated_intensities(model, group.hkl, scale=scale)
            audit_loss += observation.nll(np.r_[intensity, group.audit_background])
    result = {"schema_version": "ariadne.count-structure/1", "experimental": True,
            "success": success, "message": message, "structure": model.to_dict(), "scale": scale,
            "parameters": values[:nbase].tolist(), "parameter_names": parameterization.names,
            "initial_half_deviance": initial, "work_half_deviance": final, "audit_nll": audit_loss,
            "gradient_inf_norm": float(np.max(np.abs(gradient), initial=0.)), "maximum_background_kkt": kkt,
            "assumptions": "Calibrated independent Poisson counts, fixed profiles and geometry, fixed component scales"}
    if sections is not None:
        result["refined_profiles"] = {key: {"shift": values[section][:-1].tolist(),
                                             "width_ratio": float(np.exp(values[section][-1]))}
                                      for key, section in sections.items()}
        result["assumptions"] = "Calibrated Poisson counts, separable Gaussian profiles with shared local shifts and widths, fixed component scales"
    return result
