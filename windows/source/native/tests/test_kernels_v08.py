"""Independent optional-kernel accuracy, cancellation and buffer-contract checks."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
import json

import gemmi
import numpy as np
import pytest

from ariadne_xtal import kernels
from ariadne_xtal.derivatives import DiffractionPredictor
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.refinement import RefinementOptions, _Parameterization
from ariadne_xtal.scattering import calculated_intensities


CPP = pytest.mark.skipif(not kernels.kernel_status()["compiled_available"], reason="optional C++ extension not built")


def _problem():
    model = Structure(Cell(8.1, 8.1, 10.3), [
        Atom("Zn1", "Zn", [.123, .273, .319], f_prime=-.6, f_double_prime=2.3,
             u_aniso=np.array([[.023, .002, -.003], [.002, .031, .004], [-.003, .004, .027]])),
        Atom("O1", "O", [.329, .417, .531], u_iso=.036),
        Atom("C1", "C", [.253, .571, .387], u_iso=.024),
    ])
    laws = [np.diag([-1, -1, 1]), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])]
    options = RefinementOptions(twin_laws=[law.tolist() for law in laws], twin_fractions=[.55, .3, .15],
                                refine_twin_fractions=True)
    parameterization = _Parameterization(model, options, 1.7, np.array([.55, .3, .15]), [])
    hkl = np.array([[h, k, l] for h in range(-2, 3) for k in range(-2, 3) for l in range(-2, 3)], dtype=int)
    return parameterization, np.asarray(parameterization.start), hkl, laws


@CPP
@pytest.mark.parametrize("block_size", [1, 17, 1000])
def test_cpp_and_numpy_predict_the_same_intensities_and_derivatives(block_size):
    parameterization, values, hkl, laws = _problem()
    reference = DiffractionPredictor(parameterization, hkl, laws, backend="python").evaluate(values)
    cpp = DiffractionPredictor(parameterization, hkl, laws, block_size=block_size, backend="cpp")
    actual = cpp.evaluate(values)
    assert cpp.numeric_backend == "cpp"
    assert np.allclose(actual[1], reference[1], rtol=3e-13, atol=2e-10)
    scale = np.maximum(np.max(np.abs(reference[2]), axis=0), 1.)
    assert np.max(np.abs(actual[2]-reference[2])/scale) < 3e-13
    values = values.copy()
    values[1] += .0001
    no_jac = cpp.evaluate(values, jacobian=False)
    physical, intensity_scale, fractions = parameterization.unpack(values)
    assert np.allclose(no_jac[1], calculated_intensities(physical, hkl, intensity_scale, laws, fractions),
                       rtol=3e-13, atol=2e-10)
    assert no_jac[2] is None


def test_selector_has_a_real_reference_fallback_and_an_explicit_failure(monkeypatch):
    parameterization, values, hkl, laws = _problem()
    monkeypatch.setenv("ARIADNE_NUMERIC_BACKEND", "python")
    reference = DiffractionPredictor(parameterization, hkl, laws)
    assert reference.numeric_backend == "python"
    monkeypatch.setattr(kernels, "_cpp", None)
    monkeypatch.setattr(kernels, "_import_error", "test: extension absent")
    monkeypatch.setenv("ARIADNE_NUMERIC_BACKEND", "auto")
    fallback = DiffractionPredictor(parameterization, hkl, laws)
    assert fallback.numeric_backend == "python"
    assert np.array_equal(fallback.evaluate(values)[1], reference.evaluate(values)[1])
    assert kernels.kernel_status()["fallback_reason"] == "test: extension absent"
    with pytest.raises(RuntimeError, match="unavailable"):
        DiffractionPredictor(parameterization, hkl, laws, backend="cpp")
    with pytest.raises(ValueError, match="backend"):
        DiffractionPredictor(parameterization, hkl, laws, backend="gpu_guess")


def _raw_buffers(n=3):
    # One atom, one occupancy direction. All buffers are native and contiguous.
    hkl = np.zeros((n, 3), np.int64)
    q = np.zeros((n, 3))
    ff = np.full((1, n), 2.)
    sites = np.zeros((1, 12)); sites[:, 9] = .75
    element = np.zeros(1, np.int64)
    pointers = np.array([0, 1], np.int64)
    columns = np.zeros(1, np.int64)
    tangent = np.zeros((1, 10)); tangent[:, 9] = 1.
    output = np.full(n, -999.)
    derivative = np.full((n, 1), -999.)
    return [hkl, q, ff, sites, element, pointers, columns, tangent, output, derivative]


@CPP
def test_checked_buffer_contract_rejects_bad_views_shapes_and_aliases():
    base = _raw_buffers()
    kernels._cpp.intensity_block(*base)
    assert np.array_equal(base[-2], np.full(3, 2.25))
    assert np.array_equal(base[-1], np.full((3, 1), 6.))
    changes = []
    values = _raw_buffers(); values[1] = np.zeros((3, 6))[:, ::2]; changes.append(values)
    values = _raw_buffers(); values[1] = values[1].astype(np.float32); changes.append(values)
    values = _raw_buffers(); values[1] = values[1].astype(">f8"); changes.append(values)
    values = _raw_buffers(); values[6][0] = 1; changes.append(values)
    values = _raw_buffers(); values[5][0] = 1; changes.append(values)
    values = _raw_buffers(); values[4][0] = 10; changes.append(values)
    values = _raw_buffers(); values[0][0, 0] = 2**40; changes.append(values)
    values = _raw_buffers(); values[1][0, 0] = np.nan; changes.append(values)
    values = _raw_buffers(); values[3][0, 9] = 1.1; changes.append(values)
    values = _raw_buffers(); values[-2] = values[1].ravel()[:3]; changes.append(values)
    values = _raw_buffers(); values[-1] = np.empty((3, 2, 1)); changes.append(values)
    values = _raw_buffers(); values[-2].flags.writeable = False; changes.append(values)
    values = _raw_buffers()
    values[1] = np.ndarray((3, 3), dtype=float, buffer=bytearray(73), offset=1)
    changes.append(values)
    for values in changes:
        with pytest.raises((ValueError, TypeError, BufferError)):
            kernels._cpp.intensity_block(*values)


@CPP
def test_wrapper_copies_noncontiguous_unaligned_views_and_keeps_inputs_immutable():
    values = _raw_buffers(n=6)
    sites = kernels.PackedSites(values[3], values[4], values[5], values[6], values[7])
    unaligned = np.ndarray((6, 3), dtype=float, buffer=bytearray(145), offset=1)
    unaligned[:] = 0
    hkl = values[0][::2]
    before = [x.copy() for x in (hkl, unaligned, values[2])]
    power, jacobian = kernels.cpp_intensity_block(hkl, unaligned[::2], values[2][:, ::2], sites, 1)
    assert np.array_equal(power, np.full(3, 2.25))
    assert np.array_equal(jacobian, np.full((3, 1), 6.))
    assert all(np.array_equal(a, b) for a, b in zip(before, (hkl, unaligned, values[2])))


@CPP
def test_empty_block_with_large_parameter_dimension_does_not_allocate_scratch():
    # A zero-sized ndarray can legally carry a very large unused dimension.
    # Its C++ handling must not attempt a corresponding vector allocation.
    values = [np.empty((0, 3), np.int64), np.empty((0, 3)), np.empty((0, 0)),
              np.empty((0, 12)), np.empty(0, np.int64), np.array([0], np.int64),
              np.empty(0, np.int64), np.empty((0, 10)), np.empty(0), np.empty((0, 2**50))]
    kernels._cpp.intensity_block(*values)


@CPP
def test_compensated_structure_factor_and_derivative_preserve_a_small_difference():
    # An arithmetic stress case, not a claimed experimental crystal. Decimal
    # supplies an independent 80-digit reference; trigonometric error is zero.
    amplitudes = [1e16, 1., -1e16, 1.]
    with localcontext() as context:
        context.prec = 80
        total = sum(Decimal.from_float(v) for v in amplitudes)
        expected_intensity = float(total*total)
        expected_derivative = float(2*total*total)
    sites = np.zeros((4, 12)); sites[:, 9] = 1.; sites[:, 10] = amplitudes
    tangents = np.zeros((4, 10)); tangents[:, 9] = 1.
    output, derivative = np.empty(1), np.empty((1, 1))
    kernels._cpp.intensity_block(np.zeros((1, 3), np.int64), np.zeros((1, 3)),
        np.zeros((1, 1)), sites, np.zeros(4, np.int64), np.arange(5, dtype=np.int64),
        np.zeros(4, np.int64), tangents, output, derivative)
    assert output[0] == expected_intensity == 4.
    assert derivative[0, 0] == expected_derivative == 8.
    assert float(np.sum(np.asarray(amplitudes))) != float(total)


@CPP
def test_physical_near_extinctions_match_independent_gemmi():
    model = Structure(Cell(8.3, 9.1, 11.2), [
        Atom("Zn1", "Zn", [.123, .267, .319], u_iso=.027),
        Atom("O1", "O", [.357, .421, .537], u_iso=.031),
    ], "P 21 21 21")
    parameters = _Parameterization(model, RefinementOptions(), 1., np.ones(1), [])
    hkl = np.array([[1, 0, 0], [0, 3, 0], [0, 0, 5], [2, 0, 0], [2, 3, 1]])
    calculated = DiffractionPredictor(parameters, hkl, backend="cpp").evaluate(np.asarray(parameters.start))[1]
    independent = gemmi.SmallStructure()
    independent.cell = gemmi.UnitCell(**model.cell.to_dict())
    independent.spacegroup = gemmi.find_spacegroup_by_name(model.spacegroup)
    independent.setup_cell_images()
    for atom in model.atoms:
        site = gemmi.SmallStructure.Site()
        site.label, site.element = atom.label, gemmi.Element(atom.element)
        site.fract, site.occ, site.u_iso = gemmi.Fractional(*atom.frac), atom.occupancy, atom.u_iso
        independent.add_site(site)
    calculator = gemmi.StructureFactorCalculatorX(independent.cell)
    expected = np.array([abs(calculator.calculate_sf_from_small_structure(independent, h.tolist()))**2
                         for h in hkl])
    assert np.all(calculated[:3] < 1e-22)
    assert np.allclose(calculated, expected, rtol=4e-6, atol=1e-18)


@CPP
@pytest.mark.parametrize("occupancy", [0.0, 0.75])
def test_sparse_and_mixed_tangents_match_independent_differences(occupancy):
    # 两个位点产生干涉，覆盖坐标、ADP、占有率及混合方向。
    hkl = np.array([[1, 2, 3], [2, -1, 1], [0, 1, 2]], dtype=np.int64)
    q = hkl.astype(float) / 10
    ff = np.full((1, len(hkl)), 2.0)
    sites = np.zeros((2, 12))
    sites[:, :3] = [[.1, .2, .3], [.32, .1, .25]]
    sites[:, 3:6] = .02
    sites[:, 9] = [occupancy, 1.0]
    directions = np.zeros((5, 10))
    directions[0, :3] = [.3, .1, -.2]
    directions[1, 3:6] = [.02, .04, .06]
    directions[2, 9] = 1.0
    directions[3] = directions[:3].sum(axis=0)
    power, derivative = np.empty(len(hkl)), np.empty((len(hkl), len(directions)))
    kernels._cpp.intensity_block(hkl, q, ff, sites, np.zeros(2, np.int64),
        np.array([0, len(directions), len(directions)], np.int64),
        np.arange(len(directions), dtype=np.int64), directions, power, derivative)

    def reference(values):
        tensors = np.zeros((len(values), 3, 3))
        tensors[:, range(3), range(3)] = values[:, 3:6]
        for column, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), start=6):
            tensors[:, i, j] = tensors[:, j, i] = values[:, column]
        decay = np.exp(-2 * np.pi**2 * np.einsum("ni,sij,nj->ns", q, tensors, q))
        phase = np.exp(2j * np.pi * (hkl @ values[:, :3].T))
        return np.abs(np.sum(values[:, 9] * 2.0 * decay * phase, axis=1))**2

    expected = np.empty_like(derivative)
    for column, direction in enumerate(directions):
        plus, minus = sites.copy(), sites.copy()
        plus[0, :10] += 1e-6 * direction
        minus[0, :10] -= 1e-6 * direction
        expected[:, column] = (reference(plus) - reference(minus)) / 2e-6
    assert np.allclose(power, reference(sites), rtol=3e-14, atol=1e-14)
    assert np.allclose(derivative, expected, rtol=2e-7, atol=2e-9)
    assert np.array_equal(derivative[:, -1], np.zeros(len(hkl)))
    if occupancy == 0:
        assert np.array_equal(derivative[:, :2], np.zeros((len(hkl), 2)))
        assert np.any(np.abs(derivative[:, 2]) > .1)


@CPP
def test_sparse_tangents_preserve_nonfinite_metric_error():
    # 平方溢出不能被零方向跳过而变成静默的有效结果。
    values = _raw_buffers()
    values[1][:, 0] = 1e200
    values[3][:, 3:6] = .02
    with pytest.raises(ValueError, match="finite floating-point range"):
        kernels._cpp.intensity_block(*values)


@CPP
def test_independent_kernel_calls_are_thread_safe_without_internal_threads():
    def execute(_):
        parameterization, values, hkl, laws = _problem()
        return DiffractionPredictor(parameterization, hkl, laws, backend="cpp").evaluate(values)[1:]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, range(4)))
    for output in results[1:]:
        assert all(np.array_equal(a, b) for a, b in zip(results[0], output))
    status = kernels.kernel_status("cpp")
    assert status["build"]["internal_threads"] == 1
    assert status["build"]["releases_gil"] and not status["build"]["openmp"]
    assert not status["build"]["fast_math"]
    json.dumps(status, allow_nan=False)
