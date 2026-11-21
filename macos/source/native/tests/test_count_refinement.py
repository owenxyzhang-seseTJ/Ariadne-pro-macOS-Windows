"""Independent Poisson likelihood and structural chain-rule checks."""
import numpy as np
import pytest

from ariadne.reduction_native import PoissonObservation
from ariadne_xtal.count_refinement import CountStructureGroup, count_objective, refine_count_observations
from ariadne_xtal.derivatives import DiffractionPredictor
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.refinement import RefinementOptions, _Parameterization
from ariadne_xtal.scattering import calculated_intensities


def test_count_profile_and_gradient_match_independent_two_pixel_solution():
    model = Structure(Cell(8, 9, 10), [Atom("C1", "C", [.1, .2, .3], u_iso=.02)])
    options = RefinementOptions(scale=.17)
    parameterization = _Parameterization(model, options, options.scale, np.ones(1), [])
    values = np.asarray(parameterization.start)
    hkl = [[1, 1, 0]]
    predictor = DiffractionPredictor(parameterization, hkl)
    group = CountStructureGroup(hkl, PoissonObservation([8, 3], [[1.], [0.]], background=[[1.], [1.]]))
    score, gradient, kkt = count_objective(predictor, [group], values)
    _, power, _ = predictor.evaluate(values)
    signal = power[0]
    background = (11 - 2*signal + np.sqrt((2*signal-11)**2 + 24*signal))/4
    counts = np.array([8., 3.])
    mean = signal*np.array([1., 0.]) + background
    assert score == pytest.approx(np.sum(mean - counts + counts*np.log(counts/mean)), abs=1e-12)
    assert kkt < 1e-6
    step = 1e-5
    for j in range(len(values)):
        offset = step * np.eye(len(values))[j]
        plus = count_objective(predictor, [group], values + offset)[0]
        minus = count_objective(predictor, [group], values - offset)[0]
        assert gradient[j] == pytest.approx((plus-minus)/(2*step), abs=2e-7)


def test_count_fit_uses_signed_phase_factors_and_a_separate_audit():
    truth = Structure(Cell(8, 9, 10), [
        Atom("Zn1", "Zn", [.12, .21, .30], u_iso=.028),
        Atom("O1", "O", [.23, .49, .41], u_iso=.034),
    ])
    hkl = [(h, k, l) for h in range(1, 3) for k in range(-2, 3) for l in range(-2, 3)]
    intensity = calculated_intensities(truth, hkl, scale=2.5)
    data = np.random.default_rng(401).poisson(intensity + 3.)
    groups = [CountStructureGroup([h], PoissonObservation([n], [[1.]], offset=[3.]),
                                  role="audit" if i >= 44 else "work",
                                  audit_background=[] if i >= 44 else None)
              for i, (h, n) in enumerate(zip(hkl, data))]
    start = Structure.from_dict(truth.to_dict())
    start.atoms[1].frac += [.015, -.012, .020]
    fit = refine_count_observations(start, groups, RefinementOptions(scale=2., max_nfev=160))
    assert fit["success"]
    assert fit["work_half_deviance"] < fit["initial_half_deviance"]
    assert fit["audit_nll"] > 0
    wrong = CountStructureGroup([hkl[0]], groups[0].observation, role="audit", audit_background=[])
    with pytest.raises(ValueError, match="crosses"):
        refine_count_observations(start, [groups[0], wrong])
