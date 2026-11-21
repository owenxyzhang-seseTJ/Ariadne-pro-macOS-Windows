"""Experimental structure refinement against calibrated Poisson pixels."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .derivatives import DiffractionPredictor
from .refinement import RefinementOptions, _Parameterization, _reflection_keys, _validate_constraints
from .scattering import _indices


@dataclass
class CountStructureGroup:
    hkl: np.ndarray
    observation: object
    coefficients: np.ndarray | None = None
    role: str = "work"
    audit_background: np.ndarray | None = None

    def __post_init__(self):
        from ariadne.reduction_native import PoissonObservation

        self.hkl, _ = _indices(self.hkl)
        if isinstance(self.observation, dict):
            self.observation = PoissonObservation(**self.observation)
        if not isinstance(self.observation, PoissonObservation) or len(self.hkl) != self.observation.k:
            raise ValueError("Each count group requires one HKL per calibrated pixel profile")
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


def count_objective(predictor, groups, parameters):
    """Profile the work backgrounds and differentiate through the structure factor."""
    _, predictions, jacobian = predictor.evaluate(parameters)
    total = 0.
    gradient = np.zeros(len(parameters))
    background_kkt = 0.
    start = 0
    for group in groups:
        stop = start + len(group.hkl)
        profile = group.observation.profile_background(group.coefficients * predictions[start:stop])
        if not profile["converged"]:
            raise RuntimeError("Poisson background profile did not converge: " + profile["message"])
        score = profile["gradient"] * group.coefficients
        gradient += jacobian[start:stop].T @ score
        total += profile["half_deviance"]
        background_kkt = max(background_kkt, profile["kkt_error"])
        start = stop
    return total, gradient, background_kkt


def refine_count_observations(structure, groups, options=None, constraints=None):
    """Fit a candidate structure with fixed peak geometry and calibrated counts.

    The coefficients include fixed relative exposure, scale corrections and
    any supplied domain shares. Work background is reoptimized locally.
    Audit scores use the independently supplied background.
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
    initial, _, _ = count_objective(predictor, work, values)
    if len(values):
        fit = minimize(lambda x: count_objective(predictor, work, x)[:2], values, jac=True, method="L-BFGS-B",
                       bounds=list(zip(parameterization.lower, parameterization.upper)),
                       options={"maxfun": options.max_nfev, "maxiter": options.max_nfev,
                                "ftol": options.ftol, "gtol": options.gtol})
        values, success, message = fit.x, bool(fit.success), str(fit.message)
    else:
        success, message = True, "All structural parameters fixed"
    final, gradient, kkt = count_objective(predictor, work, values)
    model, scale, _ = parameterization.unpack(values)
    predictor.validate_site_multiplicities(model)
    audit_loss = None
    if audit:
        from .scattering import calculated_intensities

        audit_loss = sum(g.observation.nll(np.r_[g.coefficients * calculated_intensities(model, g.hkl, scale=scale),
                                                 g.audit_background]) for g in audit)
    return {"schema_version": "ariadne.count-structure/1", "experimental": True,
            "success": success, "message": message, "structure": model.to_dict(), "scale": scale,
            "parameters": values.tolist(), "parameter_names": parameterization.names,
            "initial_half_deviance": initial, "work_half_deviance": final, "audit_nll": audit_loss,
            "gradient_inf_norm": float(np.max(np.abs(gradient), initial=0.)), "maximum_background_kkt": kkt,
            "assumptions": "Calibrated independent Poisson counts, fixed profiles and geometry, fixed component scales"}
