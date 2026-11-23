import numpy as np
import pytest
from scipy.optimize import nnls

from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import calculated_intensities
from ariadne_xtal.twin_diagnostics import (candidate_rotations, partition_intensities, read_intensities,
                                          symmetry_evidence, twin_laws, twin_screen)
from ariadne_xtal.twin_phasing import direct_twin_phasing, orbit_proximal


def test_competing_twin_laws_share_predictions_and_recover_mixture():
    cell = Cell(9, 10, 11)
    structure = Structure(cell, [Atom('Fe1', 'Fe', [.12, .23, .34]), Atom('O1', 'O', [.31, .51, .14]),
                                 Atom('Cl1', 'Cl', [.85, .36, .62])])
    h = np.array(list(np.ndindex((9, 9, 9)))) - 4
    h = h[np.any(h != 0, axis=1)]
    exact = np.diag([1, -1, -1])
    calculated = calculated_intensities(structure, h, scale=1.5, twin_laws=[exact],
                                        twin_fractions=[.63, .37])
    observations = Reflections(h, calculated, np.full(len(h), 5.))
    laws = twin_laws(cell, 'P 1')
    result = twin_screen(structure, observations, laws)
    best = min(result['rows'], key=lambda row: row['chi_square'])
    assert best['fraction'] == pytest.approx(.37, abs=1e-12)
    assert best['scale'] == pytest.approx(1.5, abs=1e-12)
    assert best['fraction_interval'][0] < .37 < best['fraction_interval'][1]
    assert result['cache']['evaluated_unique_hkl'] == len(h)
    assert result['cache']['requested_hkl'] == len(h)*(1 + len(laws))


def test_absence_coverage_and_inversion_are_explicit():
    cell = Cell(11, 10, 13, 90, 99, 90)
    all_h = np.array(list(np.ndindex((7, 7, 7))))-3
    h = all_h[(all_h[:, 0] != 0) | (all_h[:, 2] != 0)]
    r = Reflections(h, np.linspace(-2, 3, len(h)), np.ones(len(h)))
    report = symmetry_evidence(cell, r, 'P 21', candidates=['P 21/m'],
                               coverage_hkl=all_h[np.any(all_h != 0, axis=1)], anomalous=True)
    alternative = next(row for row in report['candidates'] if row['spacegroup'].endswith('21/m 1'))
    assert alternative['classes']['screw_or_glide']['expected_in_declared_coverage'] > 0
    assert alternative['classes']['screw_or_glide']['unique_observed'] == 0
    assert alternative['coverage_status'] == 'equivalences_or_model_required'
    assert [row['matrix'] for row in report['twin_laws'] if row['inversion']] == [(-np.eye(3)).tolist()]
    assert report['intensity_statistics']['moments']['centric']['ideal_untwinned'] == .968


def test_half_cell_pseudotranslation_distinguishes_weak_and_missing_parity():
    cell = Cell(9, 10, 11)
    hkl = np.array(list(np.ndindex(7, 7, 7)))-3
    hkl = hkl[np.any(hkl, axis=1)]
    odd = hkl[:, 0] % 2 == 1
    reflections = Reflections(hkl, np.where(odd, 2., 100.), np.ones(len(hkl)))

    def contrast(observations):
        statistics = symmetry_evidence(cell, observations, 'P 1')['intensity_statistics']
        return next(row for row in statistics['half_cell_pseudotranslation']
                    if row['translation'] == [.5, 0., 0.])

    measured = contrast(reflections)
    assert measured['patterson_over_origin'] > .9
    assert measured['odd_mean'] < measured['even_mean']
    missing = contrast(reflections.subset(~odd))
    assert missing['status'] == 'insufficient_parity_coverage'
    assert missing['patterson_over_origin'] is None


def test_existing_reservation_survives_composed_laws():
    cell = Cell(9, 10, 11)
    h = np.array([[1, 2, 3], [1, -2, -3], *[[i, 2, 3] for i in range(2, 14)]])
    r = Reflections(h, np.ones(len(h)), np.ones(len(h)))
    laws = [{'exact': True, 'matrix': np.diag(v).tolist()} for v in ((-1, 1, -1), (-1, -1, 1))]
    roles, parts = partition_intensities(r, candidate_rotations(cell, ['P 1'], laws))
    assert roles[0] == roles[1]
    assert parts['component_ids'][0] == parts['component_ids'][1]


def test_cif_f2_reader_and_hklf5_gate(tmp_path):
    table = tmp_path/'intensity.hkl'
    table.write_text('data_observations\nloop_\n_refln_index_h\n_refln_index_k\n_refln_index_l\n'
                     '_refln_F_squared_meas\n_refln_F_squared_sigma\n 1 0 0 -5.2 2.1\n')
    assert read_intensities(table, 'cif-f2').intensity[0] == -5.2
    control = tmp_path/'input.ins'
    control.write_text('TITL twin\nHKLF 5\nEND\n')
    with pytest.raises(ValueError, match='indexed observation groups'):
        read_intensities(table, 'hklf4', control)


def test_public_hkl_with_index_only_terminator_preserves_signed_intensity(tmp_path):
    table = tmp_path/'measured.hkl'
    table.write_text('   1   2   3   -4.50    1.50   1\n   0   0   0\n', encoding='utf-8')
    data = read_intensities(table, 'hklf4')
    assert data.hkl.tolist() == [[1, 2, 3]]
    assert data.intensity.tolist() == [-4.5]
    assert data.sigma.tolist() == [1.5]


def test_reticular_law_is_partial_and_keeps_only_integral_mates():
    cell = Cell(10, 20, 15)
    h = np.array(list(np.ndindex((7, 7, 7))))-3
    h = h[np.any(h != 0, axis=1)]
    r = Reflections(h, np.ones(len(h)), np.ones(len(h)))
    law = [[0, .5, 0], [2, 0, 0], [0, 0, 1]]
    report = symmetry_evidence(cell, r, 'P 1', extra_laws=[law])
    entry, = [row for row in report['twin_laws'] if row.get('denominator') == 2]
    assert entry['route'] == 'detector_overlap'
    assert entry['sublattice_index'] == 2
    assert entry['integral_partner_indices'] == np.count_nonzero(h[:, 1] % 2 == 0)
    assert entry['metric_error'] < 1e-12


@pytest.mark.parametrize('fraction', [.0, .15, .47, .50])
def test_joint_intensity_prox_agrees_with_independent_active_set(fraction):
    rng = np.random.default_rng(75)
    target = rng.normal(0., 4., (20, 2))
    weight = rng.uniform(.25, 12., (20, 2))
    weight[::3, 1] = 0.
    power = rng.uniform(0., 2., (3, 20, 2))
    singleton = np.arange(20)%7 == 0
    result = orbit_proximal(np, power, target, weight, fraction, singleton, 4.)
    for i, trial in enumerate(power):
        for j in range(len(target)):
            alpha = 0. if singleton[j] else fraction
            m = np.array([[1.-alpha, alpha], [alpha, 1.-alpha]])
            root = np.sqrt(weight[j])
            matrix = np.vstack((root[:, None]*m, 2*np.eye(2)))
            y = np.r_[root*target[j], 2*trial[j]]
            fit, _ = nnls(matrix, y)
            np.testing.assert_allclose(result[i, j], fit, atol=2e-12, rtol=2e-12)


def test_direct_phasing_uses_work_only_and_does_not_fill_missing_observations():
    cell = Cell(11, 12, 13)
    random = np.random.default_rng(37)
    h = np.unique(random.integers(-5, 6, (140, 3)), axis=0)
    h = h[np.any(h != 0, axis=1)]
    r = Reflections(h, random.normal(800, 900, len(h)), np.full(len(h), 30.))
    result = direct_twin_phasing(cell, r, ['Fe', 'O', 'O'], np.diag([1, -1, -1]),
                                 {'trials': 2, 'keep': 2, 'iterations': 8},
                                 assume_friedel=True, refine_fraction=False, fraction=.5)
    assert result['observed_orbit_rows'] < len(r)
    assert result['missing_orbit_rows'] > 0
    assert result['candidates']
    assert all(x['likelihood']['fraction'] == .5 for x in result['candidates'])
