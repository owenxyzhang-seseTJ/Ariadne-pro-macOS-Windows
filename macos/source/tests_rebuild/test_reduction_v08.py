"""Independent likelihood, uncertainty and degeneracy checks for native pixels."""

import numpy as np
import pytest
from scipy.optimize import brentq, minimize
from scipy.special import xlogy
from scipy.stats import chi2, poisson

from ariadne.reduction_native import (
    PoissonObservation, compress_gaussian, fit_poisson,
    positive_background_basis, route_region,
)


def profiles(n=101, separation=10.):
    x = np.linspace(-30, 30, n)
    p = np.stack((np.exp(-.5*((x+separation/2)/4)**2),
                  np.exp(-.5*((x-separation/2)/4)**2)), axis=1)
    return x, p/p.sum(axis=0)


def test_gaussian_compression_is_pixel_profile_likelihood_for_arbitrary_structure_predictions():
    rng = np.random.default_rng(8401)
    x, p = profiles()
    b = np.stack((np.ones(len(x)), x/30), axis=1)
    variance = 1. + rng.random(len(x))*3
    y = p @ [60, -12] + b @ [2, .3] + rng.normal(size=len(x))*np.sqrt(variance)
    observation = compress_gaussian(y, p, variance=variance, background=b, chunk_rows=7)
    whitened_b = b/np.sqrt(variance[:, None])
    for candidate in ([-3, 7], [0, 0], [350, 180], [53, -16]):
        r = (y-p@candidate)/np.sqrt(variance)
        beta = np.linalg.lstsq(whitened_b, r, rcond=None)[0]
        direct = np.sum((r-whitened_b@beta)**2)
        assert observation.chi2(candidate) == pytest.approx(direct, rel=3e-13, abs=3e-12)
    dense = np.column_stack((p, b))/np.sqrt(variance[:, None])
    expected = np.linalg.lstsq(dense, y/np.sqrt(variance), rcond=None)[0]
    fitted = observation.fit()
    assert np.allclose(fitted['intensities'], expected[:2], rtol=1e-12, atol=1e-11)
    assert np.allclose(fitted['background_coefficients_internal'], expected[2:], rtol=1e-12)
    expected_cov = np.linalg.inv(dense.T@dense)
    assert np.allclose(fitted['joint_covariance_intensity_then_background'], expected_cov, rtol=1e-12)


def test_negative_gaussian_signal_is_not_clipped_or_dropped():
    _, p = profiles(separation=30)
    y = p @ [40, -7]
    observed = compress_gaussian(y, p, variance=1.)
    assert observed.fit()['intensities'] == pytest.approx([40, -7], abs=1e-11)
    assert observed.chi2([40, -7]) < 1e-25
    assert observed.chi2([40, 0]) > 1


def test_complete_twin_overlap_retains_sum_without_inventing_separate_intensities():
    _, shape = profiles()
    p = np.repeat(shape[:, :1], 2, axis=1)
    b = np.ones((len(p), 1))
    y = 50*p[:, 0]+2
    observed = compress_gaussian(y, p, variance=1., background=b)
    assert observed.rank == 1
    assert observed.fit()['intensities'] == [None, None]
    assert observed.contrast([1, 1])['estimate'] == pytest.approx(50)
    assert observed.contrast([1, -1])['estimate'] is None
    assert observed.chi2([20, 30]) == pytest.approx(observed.chi2([100, -50]), abs=1e-24)
    assert np.allclose(observed.R@observed.nullspace.T, 0, atol=1e-15)


def test_partial_overlap_information_loss_is_reported_without_thresholding_data():
    _, well = profiles(separation=20)
    _, almost = profiles(separation=.01)
    good = compress_gaussian(np.zeros(len(well)), well, variance=1.)
    bad = compress_gaussian(np.zeros(len(almost)), almost, variance=1.)
    assert good.rank == bad.rank == 2
    assert min(good.fit()['conditional_information_ratio']) > .9
    assert max(bad.fit()['conditional_information_ratio']) < 1e-5


def test_background_confounded_columns_have_zero_observable_rank():
    p = np.ones((12, 2))
    observed = compress_gaussian(np.arange(12.), p, variance=1., background=np.ones((12, 1)))
    assert observed.rank == 0
    assert observed.R.shape == (0, 2)
    assert observed.nullspace.shape == (2, 2)
    assert observed.fit()['intensities'] == [None, None]
    assert observed.chi2([100, -34]) == pytest.approx(np.sum((np.arange(12)-5.5)**2))


def test_rank_deficient_background_is_profiled_and_masked_nonfinite_pixels_are_excluded():
    _, p = profiles()
    y = p @ [12, 40] + .7
    b = np.ones((len(y), 2))
    valid = np.ones(len(y), dtype=bool)
    valid[[1, 7, 31]] = False
    y[~valid] = np.nan
    p[~valid] = np.nan
    b[~valid] = np.nan
    result = compress_gaussian(y, p, variance=1., background=b, valid=valid, chunk_rows=11)
    assert result.background_rank == 1
    assert result.fit()['intensities'] == pytest.approx([12, 40], abs=1e-10)
    assert result.fit()['joint_covariance_intensity_then_background'] is None


def test_underdetermined_gaussian_keeps_full_nullspace():
    p = np.array([[1., 0, 1, 0], [0, 1, 0, 1]])
    result = compress_gaussian([3, 9], p, variance=1., chunk_rows=1)
    assert result.rank == 2
    assert result.nullspace.shape == (2, 4)
    assert result.contrast([1, 0, 1, 0])['estimate'] == pytest.approx(3)


def test_background_estimation_uncertainty_changes_intensity_variance():
    _, p = profiles()
    p = p[:, :1]
    b = np.ones((len(p), 1))
    known = compress_gaussian(np.zeros(len(p)), p, variance=2.)
    unknown = compress_gaussian(np.zeros(len(p)), p, variance=2., background=b)
    assert unknown.fit()['covariance'][0][0] > known.fit()['covariance'][0][0]


def test_zero_counts_have_exact_boundary_fit_and_nonzero_profile_upper_bound():
    p = np.array([[.1], [.2], [.3]])
    fit = fit_poisson([0, 0, 0], p)
    assert fit.converged
    assert fit.intensities_internal[0] == pytest.approx(0, abs=1e-12)
    interval = fit.profile_interval([1])
    assert interval['status'] == 'ok'
    assert interval['lower'] == 0
    # -2 log LR = 2*I*sum(p) for zero observed photons.
    assert interval['upper'] == pytest.approx(chi2.ppf(.95, 1)/(2*p.sum()), rel=1e-9)
    assert fit.fisher_covariance is None


def test_partial_profile_exposure_is_not_renormalized():
    p = np.array([[.1], [.2], [.1]])
    fit = fit_poisson([3, 5, 2], p)
    assert fit.converged
    assert fit.intensities_internal[0] == pytest.approx(10/.4, rel=1e-8)


def test_large_count_offset_does_not_falsely_converge_when_unit_gradient_step_rounds_away():
    # The exact MLE is n-offset. In raw count coordinates, the initial score
    # (~1e-5) is below the ULP of the initial count (1e12). Subtracting it
    # formerly reported a false zero KKT residual at a ten-sigma error.
    count, offset = 10**12, 10**7
    fit = fit_poisson([count], [[1.]], offset=[offset])
    assert fit.converged
    assert abs(fit.intensities_internal[0]-(count-offset)) < .001*np.sqrt(count)
    assert fit.half_deviance < 1e-8
    assert fit.kkt_error < 2e-7


def test_joint_poisson_fit_matches_independent_direct_optimization():
    rng = np.random.default_rng(9083)
    x, p = profiles(separation=7)
    b = positive_background_basis(x)
    truth = np.array([34., 13., 1.2, 2.1])
    design = np.column_stack((p, b))
    y = rng.poisson(design@truth)
    model = PoissonObservation(y, p, background=b, chunk_rows=13)
    fit = model.fit()
    assert fit.converged

    def direct(t):
        mu = design@t
        return float(np.sum(mu-xlogy(y, mu)))

    independent = minimize(direct, [25, 25, 1, 1], method='Nelder-Mead',
                           bounds=[(0, None)]*4,
                           options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-11})
    actual = np.r_[fit.intensities_internal, fit.background_coefficients]
    assert independent.success
    assert np.allclose(actual, independent.x, rtol=1e-5, atol=2e-4)
    assert direct(actual) == pytest.approx(independent.fun, abs=1e-9)


def test_poisson_gradient_and_profile_envelope_gradient_are_independently_differenced():
    x, p = profiles()
    b = np.ones((len(x), 1))
    y = np.random.default_rng(847).poisson(p @ [30, 10] + 1.5)
    model = PoissonObservation(y, p, background=b, chunk_rows=9)
    values = np.array([27., 19., 1.3])
    _, analytic = model.value_gradient(values)
    assert model.nll(values) == pytest.approx(-poisson.logpmf(y, model.mean(values)).sum(), rel=1e-13)
    numeric = []
    for j in range(3):
        h = 1e-5*max(values[j], 1)
        delta = np.eye(3)[j]*h
        numeric.append((model.value_gradient(values+delta)[0]-model.value_gradient(values-delta)[0])/(2*h))
    assert np.allclose(analytic, numeric, rtol=1e-7, atol=1e-8)
    profiled = model.profile_background(values[:2], include_mean=True)
    assert profiled['converged']
    assert profiled['nll'] == pytest.approx(-poisson.logpmf(y, profiled['mean']).sum(), rel=1e-13)
    for j in range(2):
        delta = np.eye(2)[j]*1e-3
        derivative = (model.profile_background(values[:2]+delta)['half_deviance']
                      - model.profile_background(values[:2]-delta)['half_deviance'])/.002
        assert derivative == pytest.approx(profiled['gradient'][j], rel=1e-6, abs=1e-7)


def test_joint_poisson_intervals_reoptimize_background_and_are_wider_than_frozen_background():
    # Signal pixel n1~Poisson(I+b), reference n2~Poisson(b).
    # At fixed I, the positive root of the background score is analytic.
    y = np.array([8., 3.])
    p = np.array([[1.], [0.]])
    b = np.ones((2, 1))
    model = PoissonObservation(y, p, background=b)
    fit = model.fit()
    assert fit.converged
    assert fit.intensities_internal[0] == pytest.approx(5., abs=2e-6)
    assert fit.background_coefficients[0] == pytest.approx(3., abs=2e-6)
    ci = fit.profile_interval([1])
    assert ci['status'] == 'ok'
    def exact_profile(i):
        beta = (11-2*i+np.sqrt((2*i-11)**2+24*i))/4
        means = np.array([i+beta, beta])
        return float(2*np.sum(means-y+xlogy(y, y/means)))
    assert exact_profile(ci['upper']) == pytest.approx(chi2.ppf(.95, 1), rel=1e-7)
    frozen = fit_poisson(y, p, offset=np.full(2, 3.))
    frozen_ci = frozen.profile_interval([1])
    assert ci['upper'] > frozen_ci['upper']


def test_full_overlap_poisson_sum_is_identifiable_but_individuals_are_not():
    p = np.array([[.2, .2], [.3, .3], [.5, .5]])
    fit = fit_poisson([3, 2, 5], p)
    assert fit.converged
    assert fit.to_dict()['intensities'] == [None, None]
    assert sum(fit.intensities_internal) == pytest.approx(10, abs=1e-7)
    assert fit.profile_interval([1, 0])['status'] == 'not_separately_identifiable'
    ci = fit.profile_interval([1, 1])
    assert ci['status'] == 'ok'
    assert 2*(ci['upper']-10+10*np.log(10/ci['upper'])) == pytest.approx(chi2.ppf(.95, 1), rel=1e-7)


def test_poisson_never_accepts_background_subtracted_negative_or_fractional_adu_as_counts():
    for data in ([-1, 2], [.2, 2]):
        with pytest.raises(ValueError, match='genuine'):
            fit_poisson(data, [[.4], [.6]])
    with pytest.raises(ValueError, match='nonnegative'):
        fit_poisson([1, 2], [[-.2], [1]])
    with pytest.raises(ValueError, match='support'):
        fit_poisson([1, 2], [[0], [1]])


def test_poisson_masks_invalid_data_and_chunk_size_does_not_change_fit():
    _, p = profiles()
    y = np.random.default_rng(992).poisson(p@[50., 20.]+2)
    y = y.astype(float)
    valid = np.ones(len(y), dtype=bool)
    valid[3] = False
    y[3], p[3] = np.nan, np.nan
    one = fit_poisson(y, p, background=np.ones((len(y), 1)), valid=valid, chunk_rows=7)
    two = fit_poisson(y, p, background=np.ones((len(y), 1)), valid=valid, chunk_rows=1000)
    assert one.converged and two.converged
    assert np.allclose(one.intensities_internal, two.intensities_internal, rtol=3e-6, atol=1e-5)
    assert one.half_deviance == pytest.approx(two.half_deviance, abs=1e-10)


def test_positive_background_basis_is_a_partition_of_unity():
    xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [.3, .7]])
    b = positive_background_basis(xy)
    assert b.shape == (5, 4)
    assert np.all(b >= 0)
    assert np.allclose(b.sum(axis=1), 1)
    assert np.allclose(b@[1, 2, 3, 4], 1+xy[:, 0]+2*xy[:, 1])


def test_routing_uses_noise_calibration_and_geometry_without_an_observed_signal_argument():
    _, p = profiles()
    noise = {'kind': 'poisson', 'source': 'independent detector calibration',
             'poisson_counts_verified': True, 'gain': 1, 'readout_variance': 0,
             'expected_counts_source': 'previous independent exposure'}
    sparse = route_region(p, noise=noise, expected_counts=np.full(len(p), .2))
    strong = route_region(p, noise=noise, expected_counts=np.full(len(p), 200.))
    assert sparse['likelihood_action'] == strong['likelihood_action'] == 'joint_poisson_background'
    assert sparse['count_regime'] == 'sparse_counts'
    assert strong['count_regime'] == 'regular_counts'
    unresolved = route_region(np.repeat(p[:, :1], 2, axis=1), noise=noise)
    assert unresolved['overlap_regime'] == 'unresolved_combinations'
    with pytest.raises(ValueError, match='verified'):
        route_region(p, noise={**noise, 'readout_variance': 1})
    with pytest.raises(ValueError, match='fixed'):
        route_region(p, noise={'kind': 'gaussian', 'source': 'unverified'}, variance=1.)
    with pytest.raises(ValueError, match='nonnegative'):
        route_region([[-1.], [2.]], noise=noise)
    with pytest.raises(ValueError, match='nonnegative'):
        route_region([[.2], [.8]], background=[[1.], [-1.]], noise=noise)


def test_noisy_adu_route_retains_the_existing_exact_convolution_backend():
    _, p = profiles()
    route = route_region(p, noise={'kind': 'poisson_gaussian', 'source': 'detector characterization',
                                  'gain': 1.7, 'readout_variance': 4.})
    assert route['likelihood_action'] == 'existing_calibrated_poisson_gaussian_backend'
