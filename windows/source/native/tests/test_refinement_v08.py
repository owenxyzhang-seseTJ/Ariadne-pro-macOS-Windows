"""Derivative and information checks independent of the optimized forward path."""

import copy
import itertools

import numpy as np
import pytest
from scipy.linalg import expm_frechet

from ariadne_xtal.derivatives import (
    DiffractionPredictor, physical_parameter_jacobian, right_singular_system,
    symmetric_exponential_differential,
)
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.refinement import (
    RefinementOptions, _Parameterization, _validate_constraints, _validate_twins, refine,
)
from ariadne_xtal.scattering import calculated_intensities


def _indices(limit=2):
    return np.asarray([h for h in itertools.product(range(-limit, limit + 1), repeat=3)
                       if h != (0, 0, 0)], dtype=int)


def _parameterization(structure, options=None, constraints=None):
    options = options or RefinementOptions()
    laws, fractions = _validate_twins(structure, options)
    parameters = _Parameterization(structure, options, 1.7, fractions,
                                  _validate_constraints(structure, constraints))
    return parameters, np.asarray(parameters.start), laws


def _case(name):
    tensor = np.array([[.027, .002, -.004], [.002, .035, .003], [-.004, .003, .022]])
    oblique = Cell(8.3, 9.1, 11.7, 81, 97, 103)
    base = Structure(oblique, [
        Atom("Zn1", "Zn", [.123, .251, .329], u_aniso=tensor,
             f_prime=-.6, f_double_prime=2.4),
        Atom("O1", "O", [.312, .417, .532], u_iso=.036),
        Atom("C1", "C", [.241, .574, .381], u_iso=.027),
    ])
    if name == "oblique_anomalous":
        return base, RefinementOptions(origin_gauge="none"), None
    if name == "special_repeated_eigenvalues":
        return Structure(Cell(8.3, 9.1, 11.7, 90, 102, 90), [
            Atom("Zn1", "Zn", [0, .251, 0], u_aniso=np.eye(3)*.03),
            Atom("O1", "O", [.312, .417, .532], u_iso=.036),
        ], "P 1 2 1"), RefinementOptions(), None
    if name == "nonsymmorphic":
        base.cell = Cell(8.3, 9.1, 11.7, 90, 102, 90)
        base = Structure(base.cell, base.atoms, "P 21/c")
        return base, RefinementOptions(), None
    if name == "occupancy_and_inversion_twin":
        base.atoms[0].refine_occupancy = True
        base.atoms[0].occupancy = .65
        base.atoms.insert(1, Atom("Co1", "Co", base.atoms[0].frac.copy(),
                                 occupancy=.35, refine_occupancy=True, u_aniso=tensor.copy()))
        constraints = [
            {"type": "equal_xyz", "atoms": ["Zn1", "Co1"]},
            {"type": "equal_adp", "atoms": ["Zn1", "Co1"]},
            {"type": "occupancy_sum", "atoms": ["Zn1", "Co1"], "target": 1.0},
        ]
        options = RefinementOptions(twin_laws=[(-np.eye(3, dtype=int)).tolist()],
                                    twin_fractions=[.72, .28], refine_twin_fractions=True)
        return base, options, constraints
    if name == "merohedral_three_domains":
        base = Structure(Cell(9, 9, 11), base.atoms)
        laws = [np.diag([-1, -1, 1]).tolist(), [[0, 1, 0], [1, 0, 0], [0, 0, 1]]]
        return base, RefinementOptions(twin_laws=laws, twin_fractions=[.55, .3, .15],
                                       refine_twin_fractions=True), None
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["oblique_anomalous", "special_repeated_eigenvalues",
                                   "nonsymmorphic", "occupancy_and_inversion_twin",
                                   "merohedral_three_domains"])
def test_analytic_predictor_against_independent_forward_differences(name):
    structure, options, constraints = _case(name)
    parameters, start, laws = _parameterization(structure, options, constraints)
    hkl = _indices()
    predictor = DiffractionPredictor(parameters, hkl, laws, block_size=17)
    model, intensity, derivative = predictor.evaluate(start)

    def reference(values):
        model, scale, fractions = parameters.unpack(values)
        return calculated_intensities(model, hkl, scale, laws, fractions)

    assert np.allclose(intensity, reference(start), rtol=3e-13, atol=2e-10)
    independent = np.empty_like(derivative)
    for column in range(len(start)):
        delta = 2e-6 * max(1.0, abs(start[column]))
        plus, minus = start.copy(), start.copy()
        plus[column] += delta
        minus[column] -= delta
        independent[:, column] = (reference(plus) - reference(minus)) / (2 * delta)
    scale = np.maximum(np.max(np.abs(independent), axis=0), 1.0)
    assert np.max(np.abs(derivative - independent) / scale) < 2e-7
    # Reflection chunking must not change the observation model.
    _, one_block, one_derivative = DiffractionPredictor(parameters, hkl, laws,
                                                       block_size=len(hkl)).evaluate(start)
    assert np.allclose(one_block, intensity, rtol=2e-14, atol=1e-10)
    assert np.allclose(one_derivative, derivative, rtol=2e-12, atol=2e-9)
    # The physical-coordinate covariance mapping has its own analytic chain.
    physical = physical_parameter_jacobian(parameters, start)
    physical_reference = np.empty_like(physical)
    for column in range(len(start)):
        delta = 2e-6 * max(1.0, abs(start[column]))
        plus, minus = start.copy(), start.copy()
        plus[column] += delta
        minus[column] -= delta
        physical_reference[:, column] = (
            parameters.physical_values(plus)[1] - parameters.physical_values(minus)[1]
        ) / (2 * delta)
    assert np.allclose(physical, physical_reference, rtol=1e-7, atol=2e-10)


def test_log_tensor_frechet_at_degenerate_and_non_degenerate_eigenvalues():
    rng = np.random.default_rng(221)
    directions = rng.normal(size=(6, 3, 3))
    directions = (directions + directions.transpose(0, 2, 1))/2
    for matrix in (np.eye(3)*np.log(.03), np.diag([-3., -3. + 1e-13, -4.]),
                   np.array([[-3., .2, -.1], [.2, -4., .3], [-.1, .3, -3.5]])):
        actual = symmetric_exponential_differential(matrix, directions)
        expected = np.asarray([expm_frechet(matrix, direction, compute_expm=False)
                               for direction in directions])
        assert np.allclose(actual, expected, rtol=3e-13, atol=2e-15)


@pytest.mark.parametrize("shape", [(500, 17), (17, 17), (7, 19)])
def test_qr_information_modes_preserve_small_singular_values_and_nullspace(shape):
    rng = np.random.default_rng(1203)
    left, _ = np.linalg.qr(rng.normal(size=(shape[0], min(shape))))
    right, _ = np.linalg.qr(rng.normal(size=(shape[1], min(shape))))
    prescribed = np.geomspace(1.0, 1e-10, min(shape))
    matrix = (left * prescribed) @ right.T
    singular, vt = right_singular_system(matrix, block_size=32)
    reference = np.linalg.svd(matrix, compute_uv=False)
    assert np.allclose(singular, reference, rtol=1e-6, atol=1e-16)
    assert vt.shape == (shape[1], shape[1])
    assert np.allclose(vt @ vt.T, np.eye(shape[1]), atol=2e-14)
    assert np.allclose(np.linalg.norm(matrix @ vt[:len(singular)].T, axis=0), singular,
                       rtol=2e-7, atol=1e-15)
    if shape[0] < shape[1]:
        assert np.linalg.norm(matrix @ vt[shape[0]:].T) < 1e-14


def test_exact_and_legacy_derivatives_agree_on_signed_refinement():
    truth, _, _ = _case("oblique_anomalous")
    # Add a reproducible weak, signed tail while retaining one common objective.
    hkl = _indices(3)
    ideal = calculated_intensities(truth, hkl, 1.3)
    sigma = np.sqrt(ideal + 100.)
    sigma[::7] = 400.0
    observed = ideal + np.random.default_rng(31).normal(0, sigma)
    data = Reflections(hkl, observed, sigma)
    start = copy.deepcopy(truth)
    start.atoms[1].frac += [.004, -.005, .003]
    start.atoms[0].u_aniso *= 1.15
    analytic = refine(start, data, RefinementOptions(jacobian="analytic", reflection_block_size=73))
    numeric = refine(start, data, RefinementOptions(jacobian="finite_difference", reflection_block_size=73))
    assert analytic.success and numeric.success
    assert np.allclose(analytic.calculated, numeric.calculated, rtol=2e-6, atol=2e-5)
    assert analytic.metrics["chi_squared"] == pytest.approx(numeric.metrics["chi_squared"], rel=2e-10)
    assert analytic.metrics["negative_intensities"] == int(np.count_nonzero(observed < 0))
    assert np.count_nonzero(observed < 0) > 0
    assert analytic.metrics["diffraction_evaluations"] < numeric.metrics["diffraction_evaluations"] / 5


def test_noisy_restraint_and_diffraction_solution_matches_numeric_mode():
    truth = Structure(Cell(8, 9, 10), [Atom("C1", "C", [.1, .2, .3]),
                                      Atom("O1", "O", [.23, .31, .39]),
                                      Atom("N1", "N", [.41, .45, .27])])
    hkl = _indices(2)
    sigma = np.full(len(hkl), 15.)
    data = Reflections(hkl, calculated_intensities(truth, hkl), sigma)
    start = copy.deepcopy(truth)
    start.atoms[1].frac += [.006, -.004, .005]
    target = np.linalg.norm(truth.cell.to_cartesian(truth.atoms[1].frac - truth.atoms[0].frac))
    restraints = [{"type": "distance", "atoms": ["C1", "O1"], "target": target, "sigma": .01}]
    exact = refine(start, data, restraints=restraints)
    numeric = refine(start, data, RefinementOptions(jacobian="finite_difference"), restraints=restraints)
    assert exact.success and numeric.success
    assert np.allclose(exact.parameters, numeric.parameters, rtol=1e-5, atol=1e-8)
    assert exact.metrics["restraint_derivative_method"] == "central_difference"


def test_predictor_cache_and_option_validation():
    model, options, constraints = _case("oblique_anomalous")
    parameters, values, laws = _parameterization(model, options, constraints)
    predictor = DiffractionPredictor(parameters, _indices(), laws)
    predictor.evaluate(values, jacobian=False)
    predictor.evaluate(values, jacobian=False)
    assert predictor.evaluations == 1
    predictor.evaluate(values)
    predictor.evaluate(values.copy())
    assert predictor.evaluations == 2 and predictor.jacobian_evaluations == 1
    for invalid in (True, 0, -1, 1.2):
        with pytest.raises(ValueError, match="block"):
            DiffractionPredictor(parameters, _indices(), laws, block_size=invalid)
        with pytest.raises(ValueError, match="block"):
            RefinementOptions(reflection_block_size=invalid).validate()
    with pytest.raises(ValueError, match="jacobian"):
        RefinementOptions(jacobian="automatic_guess").validate()


def test_discrete_multiplicity_change_requires_an_explicit_new_model():
    model = Structure(Cell(8, 9, 10), [Atom("C1", "C", [.1, .2, .3])], "P -1")
    parameters, values, laws = _parameterization(model)
    predictor = DiffractionPredictor(parameters, _indices(), laws)
    predictor.validate_site_multiplicities(model)
    collapsed = copy.deepcopy(model)
    collapsed.atoms[0].frac[:] = 0
    with pytest.raises(ValueError, match="changed site multiplicity"):
        predictor.validate_site_multiplicities(collapsed)
