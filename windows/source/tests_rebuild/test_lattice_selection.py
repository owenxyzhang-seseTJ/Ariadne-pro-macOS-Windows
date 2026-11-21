"""A single-lattice control and weak, independent synthetic second lattice."""
import numpy as np

from ariadne.lattice_selection import select_lattices


def test_weak_lattice_requires_reserved_non_coplanar_support():
    rng = np.random.default_rng(305)
    first = np.diag([.10, .13, .18])
    second = np.array([[.085, -.072, .01], [.069, .095, .034], [.025, .0, .153]])
    nuisance = np.diag([.093, .128, .167])
    equivalent = first @ np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    error = .0008
    primary = rng.integers(-7, 8, size=(400, 3)) @ first.T + rng.normal(0, error, size=(400, 3))
    secondary = rng.integers(-7, 8, size=(70, 3)) @ second.T + rng.normal(0, error, size=(70, 3))
    outliers = rng.uniform(-1, 1, size=(30, 3))
    candidates = [first, equivalent, second, nuisance]

    def run(data):
        return select_lattices(data, candidates, np.arange(len(data)) % 4 == 0, sigma=error)

    control = run(np.concatenate([primary, outliers]))
    assert control['selected'] == [0] and control['unique_candidates'] == 3
    mixture = run(np.concatenate([primary, secondary, outliers]))
    assert mixture['selected'] == [0, 2]
    assert mixture['trace'][1]['unique_supported_hkl'] >= 6
    assert .08 < mixture['membership_weights'][2] < .20
