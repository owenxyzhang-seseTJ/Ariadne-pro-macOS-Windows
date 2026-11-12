"""Bounded native solve/refine agent with three disjoint evidence roles.

The controller proposes numerical experiments, not atomic coordinates. Work
data generate hypotheses; selection data choose them; an audit set is touched
once after the choice and optional work+selection refit. Equivalents, Friedel
mates and explicitly linked overlap components remain in one role.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .model import Reflections
from .refinement import RefinementOptions, refine
from .scattering import calculated_intensities
from .symmetry import get_operations, operation_parts


def evidence_partition(reflections: Reflections, *, rotations=(), links=(), seed=1729,
                       selection_fraction=0.15, audit_fraction=0.10):
    """Return orbit/overlap-connected work/selection/audit roles.

    Existing True free flags are reserved as the audit set. Mixed flags in one
    connected component are an error. Partitioning uses indices only, never
    intensity, so weak and negative observations are not selected away.
    ``links`` contains row-index groups that share pixels or another random
    measurement. The entire connected component is inseparable evidence.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("Partition seed must be an integer")
    if (not np.isfinite([selection_fraction, audit_fraction]).all()
            or selection_fraction <= 0 or audit_fraction <= 0
            or selection_fraction + audit_fraction >= 0.6):
        raise ValueError("Positive selection/audit fractions must sum to less than 0.6")
    hkl = reflections.hkl
    n = len(hkl)
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)
    def merge(i, j):
        parent[root(j)] = root(i)
    buckets = {}
    for i, h in enumerate(hkl):
        key = min(tuple(h), tuple(-h))
        if key in buckets:
            merge(i, buckets[key])
        else:
            buckets[key] = i
    for raw in rotations:
        rotation = np.asarray(raw, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation, np.rint(rotation), atol=1e-10, rtol=0)
                or not np.isclose(abs(np.linalg.det(rotation)), 1., atol=1e-8, rtol=0)):
            raise ValueError("Evidence rotations must be integral unimodular reciprocal transformations")
        for i, mapped in enumerate(hkl @ np.rint(rotation).astype(int)):
            key = min(tuple(mapped), tuple(-mapped))
            if key in buckets:
                merge(i, buckets[key])
    for group in links:
        group = list(group)
        if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < n for i in group):
            raise ValueError("Overlap links must contain valid reflection row integers")
        for i in group[1:]:
            merge(group[0], i)
    components = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    # Canonical index ordering makes a fixed split invariant to an input-file
    # row permutation. Otherwise merely sorting an HKL file would draw a new
    # audit set and permit accidental reuse of its former observations.
    groups = sorted(components.values(), key=lambda group:
                    min(min(tuple(hkl[i]), tuple(-hkl[i])) for i in group))
    if len(groups) < 10:
        raise ValueError("Three-way validation requires at least ten independent evidence components")
    audit, available = [], []
    existing = reflections.free_flags is not None and np.any(reflections.free_flags)
    for group in groups:
        if existing:
            flags = reflections.free_flags[group]
            if np.any(flags) and not np.all(flags):
                raise ValueError("Reserved flags split one symmetry/overlap evidence component")
            (audit if np.all(flags) else available).append(group)
        else:
            available.append(group)
    rng = np.random.default_rng(seed)
    available = [available[i] for i in rng.permutation(len(available))]
    if not existing:
        count = max(1, int(round(len(groups)*audit_fraction)))
        audit, available = available[:count], available[count:]
    select_count = max(1, int(round(len(groups)*selection_fraction)))
    selection, work = available[:select_count], available[select_count:]
    if len(work) < 3 or not selection or not audit:
        raise ValueError("Existing evidence flags leave too few independent work/selection components")
    roles = np.empty(n, dtype='<U9')
    for role, subset in (("work", work), ("selection", selection), ("audit", audit)):
        for group in subset:
            roles[group] = role
    return {"roles": roles, "component_ids": np.array([root(i) for i in range(n)]),
            "seed": int(seed), "counts": {r: int(np.count_nonzero(roles == r)) for r in ("work", "selection", "audit")},
            "component_counts": {"work": len(work), "selection": len(selection), "audit": len(audit)},
            "used_existing_audit_flags": bool(existing)}


def _subset(reflections, mask, flags=None):
    return Reflections(reflections.hkl[mask], reflections.intensity[mask], reflections.sigma[mask],
                       None if reflections.batch is None else reflections.batch[mask], flags)


@dataclass
class AutoSolveOptions:
    seed: int = 1729
    selection_fraction: float = 0.15
    audit_fraction: float = 0.10
    maximum_models: int = 16
    refit_selected: bool = True
    symmetry_tolerance: float = 0.18


def solve_refine(cell, reflections, elements, *, phasing_options=None,
                 refinement_options=None, spacegroup_candidates=None, options=None):
    """Native ab initio P1/higher-symmetry hypothesis search and local refinement.

    For a twin, the phasing input must be a justified isolated-domain or
    externally estimated phaseable set. Mixed-domain observations are refined
    with the separate native observation-operator interface. This controller
    does not silently treat a twinned total as one untwinned |F|².
    """
    from .phasing import PhasingOptions, solve_p1
    from .spacegroup_search import propose_spacegroups

    options = AutoSolveOptions() if options is None else (
        AutoSolveOptions(**options) if isinstance(options, dict) else copy.deepcopy(options))
    if not isinstance(options, AutoSolveOptions):
        raise ValueError("Invalid automatic-solve options")
    if (type(options.maximum_models) is not int or not 1 <= options.maximum_models <= 64
            or type(options.refit_selected) is not bool):
        raise ValueError("maximum_models must be 1..64 and refit_selected must be boolean")
    groups = list(dict.fromkeys(["P 1"] + list(spacegroup_candidates or [])))
    if len(groups) > 16:
        raise ValueError("At most sixteen explicit symmetry hypotheses are allowed")
    rotations = []
    for name in groups:
        rotations.extend(operation_parts(op)[0] for op in get_operations(name, cell=cell))
    partition = evidence_partition(reflections, rotations=rotations, seed=options.seed,
                                   selection_fraction=options.selection_fraction,
                                   audit_fraction=options.audit_fraction)
    roles = partition["roles"]
    available = roles != "audit"
    training = _subset(reflections, available, roles[available] == "selection")
    # Audit rows are absent from both the phasing and candidate refinement API.
    phase_options = PhasingOptions(algorithm="hybrid") if phasing_options is None else (
        PhasingOptions(**phasing_options) if isinstance(phasing_options, dict) else phasing_options)
    if phase_options.trials * phase_options.iterations > 30_000:
        raise ValueError("Automatic phase retrieval exceeds 30,000 trial-iterations")
    ref_options = RefinementOptions(max_nfev=200) if refinement_options is None else (
        RefinementOptions(**refinement_options) if isinstance(refinement_options, dict) else copy.deepcopy(refinement_options))
    ref_options.validate()
    if ref_options.twin_laws:
        raise ValueError("Native phase search requires phaseable single-domain observations; use joint-refine for overlapping domains")
    ref_options.verbose = 0
    phases = solve_p1(cell, training, elements, phase_options)
    candidates, fits, diagnostics, decisions = [], [], [], []
    decisions.append({"stage": "partition", "reason": "Indices and symmetry/overlap components determine evidence roles", **partition["counts"]})

    def fit_candidate(model, provenance):
        if len(candidates) >= options.maximum_models:
            return None
        try:
            fit = refine(model, training, ref_options)
            selection = fit.metrics["free"]
            score = float(selection["sigma_rmse"]**2)
            entry = {"id": len(candidates), **provenance, "spacegroup": model.spacegroup,
                     "training_sigma_rmse": fit.metrics["sigma_rmse"],
                     "selection_sigma_rmse": selection["sigma_rmse"],
                     "selection_score": score, "numerical_convergence": fit.success,
                     "identifiable": fit.metrics["identifiable"],
                     "n_parameters": fit.metrics["n_parameters"]}
            candidates.append(entry)
            fits.append(fit)
            return fit
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            diagnostics.append({"stage": "candidate_refinement", **provenance, "error": str(error)})
            return None

    for candidate in phases.candidates:
        fit = fit_candidate(candidate["structure"], {"trial": candidate["trial"], "source": "P1 phase retrieval"})
        if fit is not None and len(groups) > 1 and len(candidates) < options.maximum_models:
            for proposal in propose_spacegroups(fit.structure, training.hkl[~training.free_flags],
                                                groups[1:], tolerance=options.symmetry_tolerance):
                diagnostics.append({k: v for k, v in proposal.items() if k != "structure"})
                if proposal["eligible"]:
                    fit_candidate(proposal["structure"], {"trial": candidate["trial"],
                                  "source": "origin-aware symmetry projection",
                                  "phase_discrepancy": proposal["phase_discrepancy"]})
    eligible = [i for i, c in enumerate(candidates) if c["numerical_convergence"]
                and c["identifiable"] and np.isfinite(c["selection_score"])]
    if not eligible:
        return {"schema_version": "ariadne.native-autosolve/1", "status": "no_converged_identifiable_model",
                "partition": partition, "candidates": candidates, "diagnostics": diagnostics,
                "audit_evaluations": 0, "decisions": decisions, "phasing": phases.to_dict()}
    chosen = min(eligible, key=lambda i: (candidates[i]["selection_score"], candidates[i]["n_parameters"]))
    fit = fits[chosen]
    decisions.append({"stage": "select", "chosen_candidate": chosen,
                      "criterion": "mean squared standardized residual on selection rows only"})
    if options.refit_selected:
        combined = _subset(reflections, available)
        warm = copy.deepcopy(ref_options)
        warm.scale = fit.scale
        refit = refine(fit.structure, combined, warm)
        if refit.success and refit.metrics["identifiable"]:
            fit = refit
            decisions.append({"stage": "refit", "data": "work plus selection", "accepted": True})
        else:
            decisions.append({"stage": "refit", "accepted": False, "reason": "retain converged selection winner"})
    audit_mask = roles == "audit"
    prediction = calculated_intensities(fit.structure, reflections.hkl[audit_mask], scale=fit.scale)
    standardized = (reflections.intensity[audit_mask] - prediction) / reflections.sigma[audit_mask]
    audit = {"n": int(audit_mask.sum()), "chi_squared": float(standardized @ standardized),
             "sigma_rmse": float(np.sqrt(np.mean(standardized**2))),
             "negative_intensities": int(np.count_nonzero(reflections.intensity[audit_mask] < 0)),
             "used_for_candidate_generation": False, "used_for_candidate_selection": False,
             "calibration": "conditional on the input sigma values and observation model"}
    decisions.append({"stage": "audit", "evaluations": 1,
                      "retry_using_audit": False, "reason": "The audit is final evidence, not a tuning signal"})
    return {"schema_version": "ariadne.native-autosolve/1", "status": "completed",
            "partition": partition, "candidates": candidates, "selected_candidate": chosen,
            "refinement": fit.to_dict(), "audit": audit, "audit_evaluations": 1,
            "decisions": decisions, "diagnostics": diagnostics,
            "phasing": {"seed": phases.seed, "history": phases.history,
                         "assumptions": phases.assumptions},
            "interpretation": "A native structural hypothesis with independent reserved-data diagnostics"}
