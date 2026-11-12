"""Structural refinement against identifiable modes of overlapping pixels."""

import copy
import itertools
import json

import numpy as np
import pytest
from scipy.special import ndtr

from ariadne.reduction_native import compress_gaussian
from ariadne_xtal import joint_refinement
from ariadne_xtal.joint_refinement import GaussianStructureGroup, refine_observations
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.refinement import RefinementOptions
from ariadne_xtal.scattering import calculated_intensities


def make_joint_case(*, all_coincident=False, inversion=False, noise=0.0, limit=3):
    """A local-refinement experiment, never an ab initio solution benchmark."""
    truth = Structure(Cell(8., 8., 10.2), [
        Atom("Zn1", "Zn", [.13, .21, .31], u_iso=.024),
        Atom("O1", "O", [.29, .37, .43], u_iso=.031),
        Atom("C1", "C", [.41, .54, .27], u_iso=.027),
    ])
    law = -np.eye(3, dtype=int) if inversion else np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    hkl = np.asarray([h for h in itertools.product(range(-limit, limit+1), repeat=3)
                      if h != (0, 0, 0) and tuple(h) < tuple(-v for v in h)], dtype=int)
    scale, fractions = 1.6, np.array([.73, .27])
    edges = np.linspace(-5., 5., 42)
    coordinates = (edges[:-1]+edges[1:])/2
    background = np.column_stack((np.ones(len(coordinates)), coordinates/5))
    variance = 1.5+.15*coordinates**2
    rng = np.random.default_rng(541)
    groups, raw, observations = [], [], []
    for number, h in enumerate(hkl):
        component_hkl = np.vstack((h, law @ h))
        coincident = all_coincident or number % 3 == 0
        centres = [-.2, -.2 if coincident else .65]
        profiles = np.column_stack([np.diff(ndtr((edges-centre)/.8)) for centre in centres])
        intensity = calculated_intensities(truth, component_hkl, scale=scale)
        components = fractions*intensity
        data = profiles @ components + background @ np.array([2.+number*.001, .1])
        data += noise*rng.normal(0., np.sqrt(variance))
        observation = compress_gaussian(data, profiles, variance=variance, background=background,
                                        chunk_rows=11)
        group = GaussianStructureGroup.from_observation(observation, component_hkl,
                                                       domain_ids=[0, 1], label=f"region-{number}")
        groups.append(group)
        observations.append(observation)
        raw.append({"data": data, "profiles": profiles, "background": background,
                    "variance": variance, "hkl": component_hkl})
    start = copy.deepcopy(truth)
    start.atoms[1].frac += [.007, -.005, .004]
    start.atoms[2].frac += [-.004, .006, -.003]
    for atom in start.atoms:
        atom.u_iso *= 1.12
    return {"truth": truth, "start": start, "groups": groups, "raw": raw,
            "observations": observations, "scale": scale, "fractions": fractions,
            "law": law}


def raw_profiled_objective(case, structure, scale, fractions):
    score = 0.
    for region in case["raw"]:
        components = fractions*calculated_intensities(structure, region["hkl"], scale=scale)
        sigma = np.sqrt(region["variance"])
        residual = (region["data"]-region["profiles"] @ components)/sigma
        basis = region["background"]/sigma[:, None]
        beta = np.linalg.lstsq(basis, residual, rcond=None)[0]
        difference = residual-basis @ beta
        score += float(difference @ difference)
    return score


@pytest.mark.parametrize("all_coincident", [False, True])
def test_local_structure_and_domain_recovery_without_detwinning(all_coincident):
    case = make_joint_case(all_coincident=all_coincident)
    result = refine_observations(case["start"], case["groups"], domain_fractions=[.55, .45],
                                 refine_domain_fractions=True, phase_relation="same_phase_supported")
    assert result["success"] and result["identifiable"]
    assert result["domain_fractions"] == pytest.approx(case["fractions"], abs=2e-8)
    assert result["domain_fraction_estimates"] == pytest.approx(case["fractions"], abs=2e-8)
    assert result["domain_fraction_status"] == ["estimable", "estimable"]
    assert result["scale"] == pytest.approx(case["scale"], abs=2e-8)
    recovered = Structure.from_dict(result["structure"])
    for actual, expected in zip(recovered.atoms, case["truth"].atoms):
        assert np.allclose(actual.frac, expected.frac, atol=2e-8)
        assert actual.u_iso == pytest.approx(expected.u_iso, abs=2e-8)
    assert result["objective"] < 1e-12
    assert result["component_intensities_were_detwinned"] is False
    if all_coincident:
        assert all(observation.rank == 1 for observation in case["observations"])
        assert all(observation.fit()["covariance"] is None for observation in case["observations"])
    else:
        assert {observation.rank for observation in case["observations"]} == {1, 2}


def test_compressed_objective_preserves_raw_pixels_for_wrong_and_refined_models():
    case = make_joint_case(noise=1., limit=2)
    for model, scale, fractions in ((case["truth"], case["scale"], case["fractions"]),
                                     (case["start"], 1.32, np.array([.57, .43]))):
        compressed = 0.
        for group in case["groups"]:
            prediction = fractions*calculated_intensities(model, group.hkl, scale=scale)
            residual = group.R @ prediction-group.z
            compressed += residual @ residual+group.residual_constant
        raw = raw_profiled_objective(case, model, scale, fractions)
        assert compressed == pytest.approx(raw, rel=2e-12, abs=1e-9)
    result = refine_observations(case["start"], case["groups"], domain_fractions=[.55, .45],
                                 refine_domain_fractions=True, phase_relation="same_phase_supported")
    raw = raw_profiled_objective(case, Structure.from_dict(result["structure"]), result["scale"],
                                 np.asarray(result["domain_fractions"]))
    assert result["objective"] == pytest.approx(raw, rel=2e-12, abs=1e-9)
    assert result["scores"]["work"]["chi_squared"] == pytest.approx(raw, rel=2e-12, abs=1e-9)


def test_joint_chain_rule_against_function_differences(monkeypatch):
    case = make_joint_case(limit=2)
    original = joint_refinement.least_squares
    comparisons = []

    def checked_solver(fun, x0, *, jac, **kwargs):
        analytic = jac(x0)
        numeric = np.empty_like(analytic)
        for j in range(len(x0)):
            step = 2e-6*max(1., abs(x0[j]))
            plus, minus = x0.copy(), x0.copy()
            plus[j] += step
            minus[j] -= step
            numeric[:, j] = (fun(plus)-fun(minus))/(2*step)
        error = np.max(np.abs(analytic-numeric)/np.maximum(np.max(np.abs(numeric), axis=0), 1.))
        comparisons.append(float(error))
        assert error < 2e-7
        return original(fun, x0, jac=jac, **kwargs)

    monkeypatch.setattr(joint_refinement, "least_squares", checked_solver)
    result = refine_observations(case["start"], case["groups"], domain_fractions=[.55, .45],
                                 refine_domain_fractions=True, phase_relation="same_phase_supported")
    assert result["success"] and len(comparisons) == 1


def test_numerical_comparison_mode_converges_to_the_same_joint_objective():
    case = make_joint_case(limit=2)
    shared = dict(domain_fractions=[.55, .45], refine_domain_fractions=True,
                  phase_relation="same_phase_supported")
    analytic = refine_observations(case["start"], case["groups"], **shared)
    numeric = refine_observations(case["start"], case["groups"],
                                  RefinementOptions(jacobian="finite_difference"), **shared)
    assert analytic["success"] and numeric["success"]
    assert analytic["parameters"] == pytest.approx(numeric["parameters"], abs=2e-8)
    assert numeric["diffraction_derivative_method"] == "finite_difference"


def test_unobservable_inversion_fraction_is_not_assigned_a_covariance():
    case = make_joint_case(all_coincident=True, inversion=True, limit=2)
    result = refine_observations(case["start"], case["groups"], domain_fractions=[.55, .45],
                                 refine_domain_fractions=True, phase_relation="same_phase_supported")
    assert result["success"]
    assert not result["identifiable"] and result["covariance"] is None
    assert result["domain_fraction_estimates"] == [None, None]
    assert result["domain_fraction_status"] == ["unidentifiable", "unidentifiable"]
    assert result["jacobian_rank"] < result["n_parameters"]
    domain_column = result["parameter_names"].index("domain_logit[1]")
    nullspace = np.asarray(result["nullspace"])
    assert np.linalg.norm(nullspace[:, domain_column]) > .99
    assert result["objective"] < 1e-12


def test_audit_observations_do_not_change_fitted_parameters():
    case = make_joint_case(limit=2)
    audit_hkl = np.array([[8, 9, 11], [9, 8, 11]])
    audit = GaussianStructureGroup(audit_hkl, np.eye(2), [100, 200], role="audit", domain_ids=[0, 1])
    options = dict(domain_fractions=[.55, .45], refine_domain_fractions=True,
                   phase_relation="same_phase_supported")
    one = refine_observations(case["start"], case["groups"]+[audit], **options)
    changed = copy.deepcopy(audit)
    changed.z += 1e6
    two = refine_observations(case["start"], case["groups"]+[changed], **options)
    assert one["parameters"] == two["parameters"]
    assert one["scores"]["audit"]["chi_squared"] < two["scores"]["audit"]["chi_squared"]
    leaked = copy.deepcopy(case["groups"][0])
    leaked.role = "audit"
    with pytest.raises(ValueError, match="crosses"):
        refine_observations(case["start"], case["groups"]+[leaked], **options)


def test_zero_information_group_roundtrips_json_with_residual_constant():
    group = GaussianStructureGroup([[1, 2, 3], [2, 1, 3]], np.empty((0, 2)), [], 7.2,
                                    valid_pixels=10, background_rank=2)
    restored = GaussianStructureGroup(**json.loads(json.dumps(group.to_dict())))
    assert restored.R.shape == (0, 2)
    assert restored.residual_constant == 7.2


@pytest.mark.parametrize("field,value", [("domain_ids", [0, 1000000000]),
                                          ("domain_ids", [-1, 0]),
                                          ("domain_ids", [0, .5]),
                                          ("hkl", [[2**40, 0, 0], [0, 1, 0]]),
                                          ("valid_pixels", 1)])
def test_component_metadata_rejects_invalid_or_unbounded_inputs(field, value):
    values = dict(hkl=[[1, 2, 3], [2, 1, 3]], R=np.eye(2), z=[1, 1],
                  domain_ids=[0, 1], valid_pixels=3, background_rank=1)
    values[field] = value
    with pytest.raises(ValueError):
        GaussianStructureGroup(**values)


def test_impossible_pixel_rank_and_unused_twin_options_are_rejected():
    with pytest.raises(ValueError, match="plus background"):
        GaussianStructureGroup([[1, 2, 3]], [[1]], [1], valid_pixels=2, background_rank=2)
    model = Structure(Cell(8, 9, 10), [Atom("C1", "C", [.1, .2, .3])])
    group = GaussianStructureGroup([[1, 2, 3]], [[1]], [1])
    with pytest.raises(ValueError, match="replace"):
        refine_observations(model, [group], RefinementOptions(twin_fractions=[1.]))
