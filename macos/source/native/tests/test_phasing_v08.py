"""Independent mathematical and blind-recovery checks for hybrid P1 phasing."""
import copy
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigvals

from ariadne_xtal.direct_methods import (build_triplet_graph, intensity_amplitude_prox,
                                         tangent_update, triplet_residual)
from ariadne_xtal.hybrid_phasing import merge_friedel_intensities, resolution_band
from ariadne_xtal.model import Cell, Reflections
from ariadne_xtal.phasing import PhasingOptions, solve_p1
from ariadne_xtal.refinement import RefinementOptions, refine


def _recovery_case():
    script = Path(__file__).resolve().parents[1] / "validation" / "run_phasing_recovery.py"
    spec = importlib.util.spec_from_file_location("independent_phasing_case", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_radial_prox_is_global_minimum_against_independent_companion_eigenvalues():
    rng = np.random.default_rng(19091)
    for _ in range(120):
        amplitude = 10**rng.uniform(-2, 2)
        intensity = rng.uniform(-3, 4) * 10**rng.uniform(-2, 2)
        sigma = 10**rng.uniform(-2, 2)
        tau = 10**rng.uniform(-2, 2)
        value = float(intensity_amplitude_prox(amplitude, intensity, sigma, tau))
        b = sigma**2 / (2 * tau)
        # Companion eigenvalues are an independent cubic solution, including
        # every real stationary point and the nonnegative boundary.
        companion = np.array([[0., 0., b * amplitude], [1., 0., intensity - b], [0., 1., 0.]])
        roots = eigvals(companion)
        candidates = [0.] + [float(x.real) for x in roots if x.real >= 0 and abs(x.imag) < 1e-8]
        objective = lambda r: (r - amplitude)**2 / (2 * tau) + (r**2 - intensity)**2 / (2 * sigma**2)
        oracle = min(candidates, key=objective)
        assert value == pytest.approx(oracle, rel=2e-9, abs=2e-11)
        assert objective(value) <= objective(oracle) * (1 + 2e-12) + 2e-12


def test_radial_prox_keeps_signed_information_and_correct_limits():
    assert float(intensity_amplitude_prox(0., 3., 1., .25)) == pytest.approx(1.)
    assert float(intensity_amplitude_prox(0., 1., 1., .25)) == 0
    assert float(intensity_amplitude_prox(0., -3., 1., .25)) == 0
    negative = float(intensity_amplitude_prox(.5, -1., 1., .25))
    zero = float(intensity_amplitude_prox(.5, 0., 1., .25))
    positive = float(intensity_amplitude_prox(.5, 1., 1., .25))
    assert 0 < negative < zero < positive
    assert float(intensity_amplitude_prox(.3, 4., 1e-8, .25)) == pytest.approx(2., abs=1e-12)
    assert float(intensity_amplitude_prox(.3, -4., 1e8, .25)) == pytest.approx(.3, rel=1e-12)
    broadcast = intensity_amplitude_prox([.3, .7], [-4., 4.], 1.)
    assert broadcast.shape == (2,) and np.all(np.isfinite(broadcast))
    # Both terms of a naively evaluated cubic discriminant underflow here.
    # Its positive root is still representable and tends to sqrt(y).
    assert float(intensity_amplitude_prox(1., 1e-200, 3e-160, .25)) == pytest.approx(1e-100, rel=1e-12, abs=0.)
    value = float(intensity_amplitude_prox(1., 0., 2e-162, .25))
    assert np.isfinite(value) and 0 < value < 1e-100


@pytest.mark.parametrize("args", [([-1.], [1.], [1.]), ([1.], [np.nan], [1.]),
                                    ([1.], [1.], [0.]), ([1.], [1.], [np.inf])])
def test_radial_prox_rejects_invalid_physical_values(args):
    with pytest.raises(ValueError):
        intensity_amplitude_prox(*args)


@pytest.mark.parametrize("options", [{"triplet_strength": -1}, {"data_relaxation": 0},
                                      {"triplet_reflections": 0}, {"atomic_interval": 1.5},
                                      {"initialization": "truth"}])
def test_hybrid_options_validate_budgets_and_methods(options):
    with pytest.raises(ValueError):
        PhasingOptions(**options)


def test_signed_friedel_merge_has_exact_inverse_variance_mean_and_uncertainty():
    canonical = np.array([(i, 1, -1) for i in range(1, 13)])
    hkl = np.repeat(canonical, 3, axis=0)
    hkl[1::3] *= -1
    intensity = np.tile([-4., 2., 7.], 12)
    sigma = np.tile([1., 2., 3.], 12)
    flags = np.zeros(len(hkl), dtype=bool)
    flags[-3:] = True
    data = Reflections(hkl, intensity, sigma, free_flags=flags)
    before = copy.deepcopy(data.to_dict())
    hk, y, s, count = merge_friedel_intensities(data, scale=2.)
    weights = 1 / np.array([1., 2., 3.])**2
    expected = (np.array([-4., 2., 7.]) @ weights / weights.sum()) / 2
    assert len(hk) == 22
    assert np.all(y == pytest.approx(expected)) and expected < 0
    assert np.all(s == pytest.approx(1 / np.sqrt(weights.sum()) / 2))
    assert np.all(count == 3)
    assert np.array_equal(hk[::2], -hk[1::2])
    assert data.to_dict() == before
    bad = copy.deepcopy(data)
    bad.free_flags[0] = True
    with pytest.raises(ValueError, match="Holdout leakage"):
        merge_friedel_intensities(bad)


def test_triplet_graph_and_tangent_are_origin_covariant_and_hermitian():
    hkl = np.array([h for h in itertools.product(range(-3, 4), repeat=3) if h != (0, 0, 0)])
    rng = np.random.default_rng(72)
    lookup = {tuple(h): i for i, h in enumerate(hkl)}
    mate = np.array([lookup[tuple(-h)] for h in hkl])
    amplitude = rng.uniform(.3, 3, len(hkl))
    amplitude = (amplitude + amplitude[mate]) / 2
    phase = rng.uniform(-np.pi, np.pi, len(hkl))
    phase = (phase - phase[mate]) / 2
    graph = build_triplet_graph(hkl, amplitude, maximum_reflections=120, per_reflection=12)
    assert 0 < len(graph.target) <= 60 * 12
    assert np.array_equal(hkl[graph.target], hkl[graph.left] + hkl[graph.right])
    shifted = phase + 2 * np.pi * (hkl @ np.array([.139, .272, .413]))
    a = tangent_update(phase, graph, .5)
    b = tangent_update(shifted, graph, .5)
    assert np.allclose(np.exp(1j * b), np.exp(1j * (a + shifted - phase)), atol=2e-14)
    assert np.allclose(np.exp(1j * a), np.exp(-1j * a[mate]), atol=1e-14)
    assert triplet_residual(phase, graph) == pytest.approx(triplet_residual(shifted, graph), abs=1e-14)
    atom_phase = 2 * np.pi * (hkl @ np.array([.18, .36, .57]))
    assert triplet_residual(atom_phase, graph) == pytest.approx(0., abs=1e-14)


def test_resolution_band_matches_metric_and_preserves_reality_in_skew_cell():
    cell = Cell(4., 5., 6., 67., 83., 54.)
    shape = (16, 18, 20)
    limit = .31
    band = resolution_band(cell, shape, limit)
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in shape]
    hkl = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    expected = cell.stol_sq(hkl) <= limit * (1 + 1e-9)
    for axis, size in enumerate(shape):
        plane = [slice(None)] * 3
        plane[axis] = size // 2
        expected[tuple(plane)] = False
    assert np.array_equal(band, expected)
    negative = np.ix_(*[(-np.arange(n)) % n for n in shape])
    assert np.array_equal(band, band[negative])


def test_hybrid_does_not_use_holdout_values_or_mutate_observations():
    module = _recovery_case()
    truth, data = module.make_case()
    # A deterministic Friedel-invariant subset, independent of observed I.
    flags = (np.abs(data.hkl[:, 0]) + np.abs(data.hkl[:, 1]) + np.abs(data.hkl[:, 2])) % 5 == 0
    data = Reflections(data.hkl, data.intensity, data.sigma, free_flags=flags)
    altered = copy.deepcopy(data)
    altered.intensity[flags] = -1e9
    altered.sigma[flags] = 1e-9
    before = data.to_dict()
    options = PhasingOptions(trials=1, iterations=8, keep=1, algorithm="hybrid")
    a = solve_p1(truth.cell, data, ["C"] * 4, options)
    b = solve_p1(truth.cell, altered, ["C"] * 4, options)
    assert a.to_dict() == b.to_dict()
    assert data.to_dict() == before


def test_hybrid_phases_do_not_depend_on_arbitrary_intensity_units():
    module = _recovery_case()
    truth, data = module.make_case()
    options = PhasingOptions(trials=2, iterations=40, keep=2, algorithm="hybrid")
    first = solve_p1(truth.cell, data, ["C"] * 4, options)
    multiplier = 12345.
    scaled = Reflections(data.hkl, data.intensity * multiplier, data.sigma * multiplier)
    second = solve_p1(truth.cell, scaled, ["C"] * 4, options)
    by_trial = {candidate["trial"]: candidate for candidate in second.candidates}
    for a in first.candidates:
        b = by_trial[a["trial"]]
        assert a["atomic_sigma_rmse"] == pytest.approx(b["atomic_sigma_rmse"], rel=1e-8, abs=1e-9)
        assert a["profiled_scale"] * multiplier == pytest.approx(b["profiled_scale"], rel=1e-8)
        xyz_a = np.array([atom.frac for atom in a["structure"].atoms])
        xyz_b = np.array([atom.frac for atom in b["structure"].atoms])
        assert np.allclose(xyz_a, xyz_b, atol=1e-8)


def test_hybrid_blind_recovery_then_signed_f_squared_refinement():
    module = _recovery_case()
    truth, clean = module.make_case()
    rng = np.random.default_rng(142)
    sigma = np.sqrt(.8**2 + (.06 * clean.intensity)**2)
    observations = Reflections(clean.hkl, clean.intensity + rng.normal(size=len(clean)) * sigma, sigma)
    before = observations.to_dict()
    result = solve_p1(truth.cell, observations, ["C"] * 4,
                      PhasingOptions(algorithm="hybrid", trials=4, iterations=400, keep=4))
    assert len(result.candidates) == 4
    assert result.diagnostics["negative_merged_work_intensities"] > 0
    assert result.diagnostics["triplet_graph"]["triplet_count"] > 0
    assert any(c["initialization"] == "patterson_minimum_superposition" for c in result.candidates)
    # The solver receives only cell, intensities, errors and composition.
    # The truth is used solely below, for assessment modulo unavoidable gauges.
    for candidate in result.candidates:
        assert module.match_rms(truth, candidate["structure"])["rms_cartesian_angstrom"] < .04
    fit = refine(result.candidates[0]["structure"], observations, RefinementOptions(max_nfev=100))
    assert fit.success and fit.metrics["identifiable"]
    assert module.match_rms(truth, fit.structure)["rms_cartesian_angstrom"] < .004
    assert fit.metrics["negative_intensities"] == int(np.count_nonzero(observations.intensity < 0))
    assert observations.to_dict() == before
