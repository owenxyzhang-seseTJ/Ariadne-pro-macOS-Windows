"""Native structural search driven by undetwinned Gaussian pixel evidence.

Isolated work observations initialize phases.  Every work group, including
weak and overlapping components, then constrains the same atomic model
through its compressed likelihood.  The controller never fabricates a
detwinned observation table.  It preserves one frozen independent audit set.
"""
from __future__ import annotations

import copy

import numpy as np

from .autonomy import AutoSolveOptions, evidence_partition
from .joint_refinement import GaussianStructureGroup, refine_observations
from .model import Reflections, Structure
from .phasing import PhasingOptions, solve_p1
from .refinement import RefinementOptions, _json_value
from .scattering import calculated_intensities
from .spacegroup_search import propose_spacegroups
from .symmetry import get_operations, operation_parts


def _partition(groups, rotations, options):
    hkl = np.concatenate([g.hkl for g in groups])
    links, cursor = [], 0
    incoming = []
    for group in groups:
        links.append(list(range(cursor, cursor + len(group.hkl))))
        incoming.extend([group.role] * len(group.hkl))
        cursor += len(group.hkl)
    incoming = np.asarray(incoming)
    declared = set(incoming)
    if "selection" in declared and declared != {"work", "selection", "audit"}:
        raise ValueError("Explicit selection roles require a complete work/selection/audit partition")
    free = incoming == "audit" if "audit" in declared else None
    # Placeholder values are not observations.  Partitioning depends solely
    # on hkl, shared-group links, requested symmetries and reserved audit flags.
    index_design = Reflections(hkl, np.zeros(len(hkl)), np.ones(len(hkl)), free_flags=free)
    partition = evidence_partition(index_design, rotations=rotations, links=links, seed=options.seed,
                                   selection_fraction=options.selection_fraction,
                                   audit_fraction=options.audit_fraction)
    if declared == {"work", "selection", "audit"}:
        assigned = {}
        for component, role in zip(partition["component_ids"], incoming):
            if component in assigned and assigned[component] != role:
                raise ValueError("Explicit roles split a symmetry/overlap-connected evidence component")
            assigned[component] = role
        partition["roles"] = incoming
        partition["counts"] = {role: int(np.count_nonzero(incoming == role))
                               for role in ("work", "selection", "audit")}
        partition["component_counts"] = {role: sum(value == role for value in assigned.values())
                                         for role in ("work", "selection", "audit")}
        if partition["component_counts"]["work"] < 3:
            raise ValueError("At least three independent work components are required")
        partition["preserved_explicit_roles"] = True
    else:
        partition["preserved_explicit_roles"] = False
    result, start = [], 0
    for group in groups:
        clone = copy.deepcopy(group)
        group_roles = partition["roles"][start:start + len(group.hkl)]
        if len(set(group_roles)) != 1:
            raise RuntimeError("A shared pixel group was incorrectly split")
        clone.role = str(group_roles[0])
        result.append(clone)
        start += len(group.hkl)
    partition["group_counts"] = {role: sum(g.role == role for g in result)
                                 for role in ("work", "selection", "audit")}
    return result, partition


def isolated_phase_observations(groups):
    r"""Extract signed estimates only from isolated work domain-0 modes.

    For one component, ``||R c I-z||²`` has scalar estimate
    ``Ihat=(R.T z)/(R.T R)/c`` and standard uncertainty
    ``1/(sqrt(R.T R)*c)``.  Its common domain-0 fraction is absorbed into
    the arbitrary phasing intensity scale.  There is no I/sigma threshold.
    Other groups remain in the joint structure likelihood, not in this
    temporary phase-initialization table.
    """
    hkl, intensity, sigma, labels = [], [], [], []
    for group in groups:
        if (group.role != "work" or len(group.hkl) != 1 or group.domain_ids[0] != 0
                or group.coefficients[0] <= 0 or not len(group.z)):
            continue
        column = group.R[:, 0]
        length = float(np.linalg.norm(column))
        if length == 0:
            continue
        # Unit-vector form avoids unnecessarily squaring an unscaled column.
        estimate = float((column / length) @ group.z / length / group.coefficients[0])
        error = float(1 / length / group.coefficients[0])
        if not np.isfinite([estimate, error]).all() or error <= 0:
            raise ValueError("An isolated group exceeds the supported numerical intensity range")
        hkl.append(group.hkl[0])
        intensity.append(estimate)
        sigma.append(error)
        labels.append(group.label)
    independent = {min(tuple(h), tuple(-h)) for h in hkl if np.any(h)}
    if len(independent) < 10:
        raise ValueError("Fewer than ten independent isolated domain-0 work reflections; supply initial_model for joint refinement")
    result = Reflections(hkl, intensity, sigma)
    return result, {"source": "isolated domain-0 work groups only", "groups": len(hkl),
                    "independent_friedel_pairs": len(independent),
                    "negative_intensity_estimates": int(np.count_nonzero(result.intensity < 0)),
                    "intensity_sign_filter_applied": False,
                    "domain0_fraction": "common unknown factor absorbed into the phasing scale",
                    "source_labels": labels}


def score_observation_groups(model, groups, *, scale=1., domain_fractions=None):
    """Evaluate a fixed model once; this function performs no optimization.

    The full profiled chi-square includes QR residual constants.  A mode RMS
    is also reported, explicitly excluding those constants.  Only when raw
    valid-pixel and background-rank counts are available for every group is
    a full profiled-data RMS calculated.
    """
    groups = list(groups)
    if not groups:
        return {"groups": 0, "compressed_modes": 0, "chi_squared": 0.,
                "mode_sigma_rmse": None, "profiled_data_sigma_rmse": None}
    ndomain = 1 + max(int(g.domain_ids.max()) for g in groups)
    alpha = np.ones(1) if domain_fractions is None else np.asarray(domain_fractions, float)
    if (alpha.ndim != 1 or len(alpha) < ndomain or not np.isfinite(alpha).all()
            or np.any(alpha < 0) or not np.isclose(alpha.sum(), 1., atol=1e-10, rtol=0)):
        raise ValueError("Scoring requires a finite domain-fraction simplex covering every group")
    prediction = calculated_intensities(model, np.concatenate([g.hkl for g in groups]), scale=scale)
    cursor, mode_chi2, constant, modes = 0, 0., 0., 0
    known_counts = all(g.valid_pixels is not None for g in groups)
    profiled_count = 0 if known_counts else None
    details = []
    for group in groups:
        base = prediction[cursor:cursor + len(group.hkl)]
        cursor += len(group.hkl)
        components = group.coefficients * alpha[group.domain_ids] * base
        residual = group.R @ components - group.z
        value = float(residual @ residual)
        mode_chi2 += value
        constant += group.residual_constant
        modes += len(group.z)
        if known_counts:
            profiled_count += group.valid_pixels - group.background_rank
        details.append({"label": group.label, "role": group.role,
                        "chi_squared": value + group.residual_constant,
                        "n_components": len(group.hkl), "n_modes": len(group.z)})
    return {"groups": len(groups), "compressed_modes": modes,
            "chi_squared": mode_chi2 + constant, "mode_chi_squared": mode_chi2,
            "residual_constant": constant,
            "mode_sigma_rmse": float(np.sqrt(mode_chi2 / modes)) if modes else None,
            "profiled_data_dimension": profiled_count,
            "profiled_data_sigma_rmse": float(np.sqrt((mode_chi2 + constant) / profiled_count))
            if profiled_count else None, "group_diagnostics": details,
            "calibration": "conditional on fixed profile geometry, calibration, background basis and Gaussian variances"}


def solve_observations(cell, elements, groups, *, initial_model=None,
                       phasing_options=None, refinement_options=None,
                       spacegroup_candidates=None, options=None,
                       domain_fractions=None, refine_domain_fractions=False,
                       phase_relation="single_phase", restraints=None, constraints=None):
    """Solve/refine one structural phase directly from mixed pixel likelihoods.

    Component hkls are expressed in their indexed domain's common structural
    cell. Explicit calibration coefficients supply known exposure, partiality,
    scale/absorption factors; global domain fractions can be fitted. Multiple
    domains require the explicit ``same_phase_supported`` declaration.

    Automatic initialization requires sufficiently many isolated domain-0
    work reflections. With ``initial_model``, that phase step is bypassed.
    No twin fraction, reflection sign, audit residual or chemically guessed
    coordinate is used to manufacture measured component intensities.
    """
    groups = [GaussianStructureGroup(**g) if isinstance(g, dict) else copy.deepcopy(g) for g in groups]
    if not groups or any(not isinstance(g, GaussianStructureGroup) for g in groups):
        raise ValueError("Provide nonempty GaussianStructureGroup evidence")
    options = AutoSolveOptions() if options is None else (
        AutoSolveOptions(**options) if isinstance(options, dict) else copy.deepcopy(options))
    if not isinstance(options, AutoSolveOptions):
        raise ValueError("Invalid automatic-solve options")
    if (type(options.maximum_models) is not int or not 1 <= options.maximum_models <= 64
            or type(options.refit_selected) is not bool
            or not np.isfinite(options.symmetry_tolerance) or options.symmetry_tolerance <= 0):
        raise ValueError("Invalid model budget, refit flag or symmetry tolerance")
    if initial_model is not None:
        initial_model = Structure.from_dict(initial_model) if isinstance(initial_model, dict) else copy.deepcopy(initial_model)
        if not isinstance(initial_model, Structure) or not np.allclose(initial_model.cell.matrix, cell.matrix,
                                                                       rtol=1e-10, atol=1e-10):
            raise ValueError("initial_model must use the supplied unit cell")
    names = list(dict.fromkeys(["P 1"] + list(spacegroup_candidates or []) +
                              ([] if initial_model is None else [initial_model.spacegroup])))
    if len(names) > 16:
        raise ValueError("At most sixteen explicit symmetry hypotheses are allowed")
    rotations = [operation_parts(op)[0] for name in names for op in get_operations(name, cell=cell)]
    groups, partition = _partition(groups, rotations, options)
    training = [g for g in groups if g.role != "audit"]
    work = [g for g in groups if g.role == "work"]
    selection = [g for g in groups if g.role == "selection"]
    audit = [g for g in groups if g.role == "audit"]
    if not any(np.any(g.R * g.coefficients[None, :]) for g in selection):
        raise ValueError("Selection evidence contains no model-dependent observable mode")
    ndomain = 1 + max(int(g.domain_ids.max()) for g in groups)
    if 1 + max(int(g.domain_ids.max()) for g in training) < ndomain:
        raise ValueError("A domain appears only in the reserved audit set and has no training calibration")
    fractions = np.full(ndomain, 1 / ndomain) if domain_fractions is None else np.asarray(domain_fractions, float)
    if (fractions.shape != (ndomain,) or not np.isfinite(fractions).all() or np.any(fractions < 0)
            or not np.isclose(fractions.sum(), 1., atol=1e-10, rtol=0)):
        raise ValueError("domain_fractions must cover every domain and sum to one")
    if phase_relation not in {"single_phase", "same_phase_supported"} or (ndomain > 1 and phase_relation != "same_phase_supported"):
        raise ValueError("Multiple indexed domains require phase_relation='same_phase_supported'")
    if type(refine_domain_fractions) is not bool:
        raise ValueError("refine_domain_fractions must be boolean")
    ref_options = RefinementOptions(max_nfev=200) if refinement_options is None else (
        RefinementOptions(**refinement_options) if isinstance(refinement_options, dict) else copy.deepcopy(refinement_options))
    ref_options.validate()
    ref_options.verbose = 0
    decisions = [{"stage": "partition", "reason": "Shared pixel groups and symmetry orbits are inseparable evidence",
                  "group_counts": partition["group_counts"]}]
    if initial_model is None:
        if fractions[0] <= 0:
            raise ValueError("Domain-0 fraction is zero; supply an initial_model instead of domain-0 phasing")
        phase_data, initialization = isolated_phase_observations(work)
        phase_options = PhasingOptions(algorithm="hybrid") if phasing_options is None else (
            PhasingOptions(**phasing_options) if isinstance(phasing_options, dict) else copy.deepcopy(phasing_options))
        if phase_options.trials * phase_options.iterations > 30_000:
            raise ValueError("Automatic phase retrieval exceeds 30,000 trial-iterations")
        phase_result = solve_p1(cell, phase_data, elements, phase_options)
        hypotheses = [(candidate["structure"], {"trial": candidate["trial"], "source": "isolated-work phase retrieval"})
                      for candidate in phase_result.candidates]
        phase_evidence = {"seed": phase_result.seed, "history": phase_result.history,
                          "diagnostics": phase_result.diagnostics, "assumptions": phase_result.assumptions}
    else:
        hypotheses = [(initial_model, {"source": "explicit initial structural model"})]
        initialization = {"source": "explicit initial structural model", "blind_phase_search_performed": False}
        phase_evidence = None
    decisions.append({"stage": "initialize", **initialization})
    candidates, fits, diagnostics = [], [], []

    def fit_hypothesis(model, provenance):
        if len(candidates) >= options.maximum_models:
            return None
        try:
            fit = refine_observations(model, training, ref_options,
                                      domain_fractions=fractions, refine_domain_fractions=refine_domain_fractions,
                                      phase_relation=phase_relation, restraints=restraints, constraints=constraints)
            score = fit["scores"]["selection"]
            entry = {"id": len(candidates), **provenance, "spacegroup": model.spacegroup,
                     "work_chi_squared": fit["scores"]["work"]["chi_squared"],
                     "selection_chi_squared": score["chi_squared"],
                     "selection_compressed_modes": score["compressed_modes"],
                     "numerical_convergence": fit["success"], "identifiable": fit["identifiable"],
                     "n_parameters": fit["n_parameters"]}
            candidates.append(entry)
            fits.append(fit)
            return fit
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            diagnostics.append({"stage": "joint_refinement", **provenance, "error": str(error)})
            return None

    work_hkl = np.concatenate([g.hkl for g in work])
    for model, provenance in hypotheses:
        fit = fit_hypothesis(model, provenance)
        if fit is None or len(candidates) >= options.maximum_models or len(names) == 1:
            continue
        fitted_model = Structure.from_dict(fit["structure"])
        if len(fitted_model.operations) != 1:
            continue
        for proposal in propose_spacegroups(fitted_model, work_hkl, names[1:], tolerance=options.symmetry_tolerance):
            diagnostics.append({k: v for k, v in proposal.items() if k != "structure"})
            if proposal["eligible"]:
                fit_hypothesis(proposal["structure"], {**provenance, "source": "origin-aware symmetry projection",
                                                       "phase_discrepancy": proposal["phase_discrepancy"]})
    eligible = [i for i, item in enumerate(candidates) if item["numerical_convergence"]
                and item["identifiable"] and np.isfinite(item["selection_chi_squared"])]
    result = {"schema_version": "ariadne.native-evidence-solve/1", "partition": partition,
              "candidates": candidates, "diagnostics": diagnostics, "decisions": decisions,
              "phasing": phase_evidence, "initialization": initialization,
              "component_intensities_were_detwinned": False}
    if not eligible:
        result.update(status="no_converged_identifiable_model", audit_evaluations=0)
        return _json_value(result)
    chosen = min(eligible, key=lambda i: (candidates[i]["selection_chi_squared"], candidates[i]["n_parameters"]))
    fit = fits[chosen]
    decisions.append({"stage": "select", "chosen_candidate": chosen,
                      "criterion": "profiled Gaussian chi-square on selection groups only"})
    if options.refit_selected:
        combined = copy.deepcopy(training)
        for group in combined:
            group.role = "work"
        warm = copy.deepcopy(ref_options)
        warm.scale = fit["scale"]
        refit = refine_observations(Structure.from_dict(fit["structure"]), combined, warm,
                                    domain_fractions=fit["domain_fractions"], refine_domain_fractions=refine_domain_fractions,
                                    phase_relation=phase_relation, restraints=restraints, constraints=constraints)
        if refit["success"] and refit["identifiable"]:
            fit = refit
            decisions.append({"stage": "refit", "data": "work plus selection groups", "accepted": True})
        else:
            decisions.append({"stage": "refit", "accepted": False, "reason": "retain converged selection winner"})
    audit_score = score_observation_groups(Structure.from_dict(fit["structure"]), audit,
                                           scale=fit["scale"], domain_fractions=fit["domain_fractions"])
    audit_score.update(used_for_candidate_generation=False, used_for_candidate_selection=False)
    decisions.append({"stage": "audit", "evaluations": 1, "retry_using_audit": False})
    result.update(status="completed", selected_candidate=chosen, refinement=fit,
                  audit=audit_score, audit_evaluations=1,
                  interpretation="Structure fitted to original group likelihoods; domain components remain predictions, not invented measurements")
    return _json_value(result)
