"""Typed JSON bridge to Ariadne's calibrated pixel likelihoods."""
from __future__ import annotations

import numpy as np


def _matrix_shape(value, name):
    """Check JSON dimensions before allocating its floating-point matrix."""
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError(f"{name} must be a matrix")
        return value.shape
    if not isinstance(value, (list, tuple)) or not value or not isinstance(value[0], (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a nonempty matrix")
    return len(value), len(value[0])


def integrate_problem(problem):
    from ariadne.reduction_native import compress_gaussian, fit_poisson, route_region
    if not isinstance(problem, dict):
        raise ValueError("Pixel problem must be an object")
    allowed = {"data", "profiles", "variance", "background", "valid", "noise", "offset",
               "expected_counts", "chunk_rows", "component_hkl", "coefficients", "domain_ids",
               "role", "label", "interval_contrasts", "confidence_level"}
    if set(problem)-allowed:
        raise ValueError(f"Unknown pixel fields: {sorted(set(problem)-allowed)}")
    rows, components = _matrix_shape(problem["profiles"], "profiles")
    if not 1 <= rows <= 1_000_000 or not 1 <= components <= 256:
        raise ValueError("Agent pixel budget is 1,000,000 pixels and 256 components per region")
    background_rows, background_columns = ((rows, 0) if problem.get("background") is None else
                                           _matrix_shape(problem["background"], "background"))
    if background_columns > 64:
        raise ValueError("Agent background budget is 64 columns per region")
    if background_rows != rows:
        raise ValueError("Background and profile matrices must describe the same pixels")
    if rows*(components+background_columns) > 8_000_000:
        raise ValueError("Agent pixel design budget is 8,000,000 profile/background entries")
    contrasts = problem.get("interval_contrasts", [])
    if not isinstance(contrasts, list) or len(contrasts) > 16:
        raise ValueError("At most sixteen declared profile-interval contrasts are allowed")
    data, profiles = np.asarray(problem["data"], float), np.asarray(problem["profiles"], float)
    if data.shape != (rows,) or profiles.shape != (rows, components):
        raise ValueError("Pixel data and profile shapes are inconsistent")
    common = {key: problem[key] for key in ("background", "valid", "chunk_rows") if key in problem}
    route = route_region(profiles, noise=problem["noise"], variance=problem.get("variance"),
                         expected_counts=problem.get("expected_counts"), **common)
    if route["likelihood_action"] == "gaussian_observable_likelihood":
        offset = np.asarray(0. if problem.get("offset") is None else problem["offset"], float)
        if offset.shape not in ((), data.shape):
            raise ValueError("Known Gaussian mean offset must be scalar or one value per pixel")
        observation = compress_gaussian(data-offset, profiles, variance=problem["variance"], **common)
        result = {"status": "completed", "route": route,
                  "observation": observation.to_dict(), "diagnostic_fit": observation.fit(),
                  "known_mean_offset_applied": problem.get("offset") is not None}
        if "component_hkl" in problem:
            from .joint_refinement import GaussianStructureGroup
            fields = {key: problem[key] for key in ("coefficients", "domain_ids", "role", "label") if key in problem}
            group = GaussianStructureGroup.from_observation(observation, problem["component_hkl"], **fields)
            result["structure_group"] = group.to_dict()
    elif route["likelihood_action"] == "joint_poisson_background":
        fit = fit_poisson(data, profiles, offset=problem.get("offset"), **common)
        result = {"status": "completed" if fit.converged else "not_converged", "route": route, "fit": fit.to_dict()}
        result["intervals"] = [fit.profile_interval(c, confidence_level=problem.get("confidence_level", .95)) for c in contrasts]
    else:
        result = {"status": "requires_calibrated_backend", "route": route,
                  "backend": "Ariadne skills/ariadne-reduction/scripts/overlap_integrator.py --estimator poisson-gaussian",
                  "reason": "Readout-noisy ADU require the retained calibrated Poisson-Gaussian convolution model"}
    return result
