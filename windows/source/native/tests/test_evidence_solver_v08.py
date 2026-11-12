"""Checks for native solving directly from compressed, overlapping evidence."""
import copy
import itertools

import numpy as np
import pytest

import ariadne_xtal.evidence_solver as solver
from ariadne_xtal.autonomy import AutoSolveOptions
from ariadne_xtal.joint_refinement import GaussianStructureGroup
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.scattering import calculated_intensities


def make_joint_case():
    cell = Cell(5.5, 5.5, 6.2)
    truth = Structure(cell, [Atom("C1", "C", [.13, .24, .35], u_iso=.012),
                             Atom("O1", "O", [.36, .62, .79], u_iso=.014),
                             Atom("N1", "N", [.71, .16, .56], u_iso=.018)])
    hkl = np.array([h for h in itertools.product(range(-4, 5), repeat=3)
                    if h != (0, 0, 0) and h < tuple(-v for v in h)])
    alpha, groups = np.array([.63, .37]), []
    for i, h in enumerate(hkl):
        if i % 5 == 0:
            hk, domains, coefficients = np.array([h, h[[1, 0, 2]]]), [0, 1], np.array([.8, 1.2])
            factor = np.array([[.5, .48], [0., .08]])
        else:
            hk, domains, coefficients = h[None, :], [1 if i % 5 == 1 else 0], np.array([1.2])
            factor = np.array([[.6]])
        target = factor @ (coefficients * alpha[domains] * calculated_intensities(truth, hk, scale=1.3))
        groups.append(GaussianStructureGroup(hk, factor, target, coefficients=coefficients,
                                             domain_ids=domains, label=str(i), valid_pixels=len(target)))
    initial = copy.deepcopy(truth)
    rng = np.random.default_rng(1)
    for atom in initial.atoms:
        atom.frac += rng.normal(0, .009, 3)
        atom.u_iso *= 1.15
    return truth, initial, groups


def test_isolated_phase_table_retains_negative_estimates_and_exact_scalar_likelihood():
    groups = []
    column = np.array([2., -1., .5])
    coefficient = 2.5
    for i in range(12):
        # The extra vector is orthogonal to R and cannot change the fitted I.
        target = column * coefficient * (i - 5) + np.array([.1, .2, 0.])
        groups.append(GaussianStructureGroup([[i + 1, 1, 0]], column[:, None], target,
                                             coefficients=[coefficient], label=str(i)))
    groups += [GaussianStructureGroup([[40, 1, 0]], [[1.]], [1e10], role="audit"),
               GaussianStructureGroup([[41, 1, 0]], [[1.]], [-1e10], role="selection"),
               GaussianStructureGroup([[42, 1, 0]], [[1.]], [1e10], domain_ids=[1]),
               GaussianStructureGroup([[43, 1, 0], [44, 1, 0]], [[1., 1.]], [1e10])]
    data, diagnostic = solver.isolated_phase_observations(groups)
    assert len(data) == 12
    assert np.allclose(data.intensity, np.arange(12) - 5)
    assert np.allclose(data.sigma, 1 / np.linalg.norm(column) / coefficient)
    assert diagnostic["negative_intensity_estimates"] == 5
    assert diagnostic["intensity_sign_filter_applied"] is False


def test_fully_overlapped_groups_do_not_invent_phaseable_component_intensities():
    groups = [GaussianStructureGroup([[i + 1, 1, 0], [1, i + 1, 0]], [[1., 1.]], [30.],
                                     domain_ids=[0, 1]) for i in range(20)]
    with pytest.raises(ValueError, match="initial_model"):
        solver.isolated_phase_observations(groups)


def test_joint_evidence_solver_recovers_domains_and_never_trains_on_audit(monkeypatch):
    truth, initial, groups = make_joint_case()
    before = [g.to_dict() for g in groups]
    partitioned, _ = solver._partition(groups, [np.eye(3)], AutoSolveOptions())
    audit_labels = {g.label for g in partitioned if g.role == "audit"}
    actual_refine = solver.refine_observations
    actual_score = solver.score_observation_groups
    calls = {"refine": 0, "audit": 0}

    def refine_spy(model, inputs, *args, **kwargs):
        assert not any(g.label in audit_labels for g in inputs)
        calls["refine"] += 1
        return actual_refine(model, inputs, *args, **kwargs)

    def score_spy(model, inputs, *args, **kwargs):
        assert {g.label for g in inputs} == audit_labels
        calls["audit"] += 1
        return actual_score(model, inputs, *args, **kwargs)

    monkeypatch.setattr(solver, "refine_observations", refine_spy)
    monkeypatch.setattr(solver, "score_observation_groups", score_spy)
    args = dict(initial_model=initial, refinement_options={"max_nfev": 100},
                domain_fractions=[.5, .5], refine_domain_fractions=True,
                phase_relation="same_phase_supported")
    result = solver.solve_observations(truth.cell, ["C", "O", "N"], groups, **args)
    assert result["status"] == "completed"
    assert result["component_intensities_were_detwinned"] is False
    assert result["initialization"]["blind_phase_search_performed"] is False
    assert result["refinement"]["domain_fractions"] == pytest.approx([.63, .37], abs=1e-9)
    assert result["refinement"]["scale"] == pytest.approx(1.3, abs=1e-9)
    assert result["audit"]["chi_squared"] < 1e-15
    assert calls == {"refine": 2, "audit": 1}
    altered = copy.deepcopy(groups)
    for group in altered:
        if group.label in audit_labels:
            group.z += 1e6
            group.R *= 3.
    changed = solver.solve_observations(truth.cell, ["C", "O", "N"], altered, **args)
    assert result["candidates"] == changed["candidates"]
    assert result["refinement"] == changed["refinement"]
    assert changed["audit"]["chi_squared"] > 1e12
    assert calls == {"refine": 4, "audit": 2}
    assert before == [g.to_dict() for g in groups]


def test_full_explicit_roles_are_preserved_and_conflicting_orbits_rejected():
    truth, _, groups = make_joint_case()
    first, partition = solver._partition(groups, [np.eye(3)], AutoSolveOptions())
    second, repeated = solver._partition(first, [np.eye(3)], AutoSolveOptions(seed=987))
    assert [g.role for g in first] == [g.role for g in second]
    assert repeated["preserved_explicit_roles"]
    assert partition["group_counts"] == repeated["group_counts"]
    # Repeat the same evidence in a different role: it must not become an
    # independent validator simply because a new group label was assigned.
    leaked = copy.deepcopy(first)
    duplicate = copy.deepcopy(next(g for g in first if g.role == "work"))
    duplicate.role = "selection"
    leaked.append(duplicate)
    with pytest.raises(ValueError, match="split"):
        solver._partition(leaked, [np.eye(3)], AutoSolveOptions())
