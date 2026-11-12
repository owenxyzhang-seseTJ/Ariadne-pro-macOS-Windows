"""Independent validation of origin search and evidence-separated solving."""
import copy
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

import ariadne_xtal.autonomy as autonomy
import ariadne_xtal.phasing as phasing
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.scattering import structure_factors
from ariadne_xtal.spacegroup_search import phase_origin_search, project_spacegroup
from ariadne_xtal.symmetry import get_operations, operation_parts, unique_sites


def _indices(limit=3):
    return np.array([h for h in itertools.product(range(-limit, limit + 1), repeat=3) if h != (0, 0, 0)])


def test_arbitrary_origin_inversion_projection_preserves_complex_structure_factors():
    cell = Cell(7., 8., 9., 82., 101., 94.)
    source = Structure(cell, [Atom("C1", "C", [.12, .23, .34], u_iso=.018),
                              Atom("O1", "O", [.27, .38, .42], f_prime=-.13, f_double_prime=.22),
                              Atom("N1", "N", [.43, .51, .26],
                                   u_aniso=np.array([[.02, .002, .003], [.002, .03, .004], [.003, .004, .04]]))],
                       "P -1")
    unknown_shift = np.array([.143, .278, .337])
    atoms = []
    for atom in source.atoms:
        for i, (position, tensor) in enumerate(unique_sites(source, atom)):
            full = copy.deepcopy(atom)
            full.label += f"_{i}"
            full.frac = (position + unknown_shift) % 1
            full.u_aniso = tensor
            atoms.append(full)
    p1 = Structure(cell, atoms)
    before = p1.to_dict()
    hkl = _indices()
    search = phase_origin_search(p1, hkl, "P -1")
    assert search["origin_rank"] == 3 and search["numerical_convergence"]
    assert search["phase_discrepancy"] < 1e-20
    projected, diagnostic = project_spacegroup(p1, "P -1", search["origin_shift"])
    assert len(projected.atoms) == 3
    assert diagnostic["unit_cell_atoms"] == 6
    phase_shift = np.exp(2j * np.pi * (hkl @ np.asarray(search["origin_shift"])))
    assert np.allclose(structure_factors(projected, hkl), structure_factors(p1, hkl) * phase_shift,
                       atol=2e-12, rtol=2e-12)
    assert p1.to_dict() == before


@pytest.mark.parametrize("spacegroup,cell", [
    ("P -1", Cell(7., 8., 9., 82., 101., 94.)),
    ("P 21", Cell(7., 8., 9., 90., 103., 90.)),
])
def test_mixed_displacement_projection_is_invariant_to_orbit_atom_order(spacegroup, cell):
    rotation, translation = next((r, t) for r, t in map(operation_parts, get_operations(spacegroup, cell=cell))
                                 if not np.array_equal(r, np.eye(3)))
    position = np.array([.11, .23, .37])
    isotropic = .022
    other_u = np.array([[.01, .002, .003], [.002, .025, -.004], [.003, -.004, .04]])
    iso_atom = Atom("C1", "C", position, u_iso=isotropic)
    # u_iso is deliberately a stale placeholder: the physical anisotropic
    # tensor, not that placeholder, defines the projected displacement.
    aniso_atom = Atom("C2", "C", rotation@position+translation, u_iso=.09, u_aniso=other_u)
    cart_rotation = cell.matrix@rotation@np.linalg.inv(cell.matrix)
    inverse = np.linalg.inv(cart_rotation)
    expected_u = (isotropic*np.eye(3) + inverse@other_u@inverse.T)/2
    target = Structure(cell, [Atom("C", "C", position, u_iso=np.trace(expected_u)/3,
                                   u_aniso=expected_u)], spacegroup)
    hkl = _indices()
    expected_f = structure_factors(target, hkl)
    for order in ([iso_atom, aniso_atom], [aniso_atom, iso_atom]):
        source = Structure(cell, copy.deepcopy(order), "P 1")
        before = source.to_dict()
        projected, diagnostic = project_spacegroup(source, spacegroup, [0., 0., 0.])
        assert len(projected.atoms) == 1 and diagnostic['unit_cell_atoms'] == 2
        atom = projected.atoms[0]
        assert atom.u_aniso is not None
        assert atom.u_iso == pytest.approx(np.trace(atom.u_aniso)/3, rel=1e-14)
        # Compare full complex amplitudes against the independently constructed
        # inverse-rotated tensor mean, not against the nonsymmetric input model.
        assert np.allclose(structure_factors(projected, hkl), expected_f, rtol=2e-12, atol=2e-12)
        assert source.to_dict() == before


def test_evidence_roles_group_rotations_friedel_and_overlap_links_without_intensity_selection():
    hkl = _indices(3)
    data = Reflections(hkl, np.arange(len(hkl)) - 100., np.ones(len(hkl)))
    rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    links = [[4, 42, 108], [108, 201]]
    first = autonomy.evidence_partition(data, rotations=[rotation], links=links, seed=87)
    by_h = {tuple(h): role for h, role in zip(hkl, first["roles"])}
    for h in hkl:
        assert by_h[tuple(h)] == by_h[tuple(-h)] == by_h[tuple(h @ rotation)]
    for group in links:
        assert len(set(first["roles"][group])) == 1
    # Evidence roles must survive intensity changes, file reordering and
    # a corresponding reindexing of the explicitly declared overlap links.
    rng = np.random.default_rng(77)
    order = rng.permutation(len(data))
    inverse = np.argsort(order)
    altered = Reflections(hkl[order], rng.normal(size=len(data)) * 1e8, np.full(len(data), 1e3))
    second = autonomy.evidence_partition(altered, rotations=[rotation],
                                         links=[[int(inverse[i]) for i in group] for group in links], seed=87)
    assert np.array_equal(first["roles"][order], second["roles"])
    assert set(first["roles"]) == {"work", "selection", "audit"}
    with pytest.raises(ValueError, match="unimodular"):
        autonomy.evidence_partition(data, rotations=[np.zeros((3, 3))])


def test_conflicting_reserved_flags_cannot_split_an_evidence_component():
    hkl = _indices(2)
    flags = np.zeros(len(hkl), dtype=bool)
    flags[0] = True
    data = Reflections(hkl, np.ones(len(hkl)), np.ones(len(hkl)), free_flags=flags)
    with pytest.raises(ValueError, match="split"):
        autonomy.evidence_partition(data)


def test_actual_blind_solve_refine_keeps_audit_absent_until_one_final_evaluation(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "validation" / "run_phasing_recovery.py"
    spec = importlib.util.spec_from_file_location("autonomous_blind_case", path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    truth, clean = case.make_case()
    rng = np.random.default_rng(142)
    sigma = np.sqrt(.8**2 + (.06 * clean.intensity)**2)
    data = Reflections(clean.hkl, clean.intensity + rng.normal(size=len(clean)) * sigma, sigma)
    before = data.to_dict()
    partition = autonomy.evidence_partition(data, seed=1729)
    audit_mask = partition["roles"] == "audit"
    audit_indices = {tuple(h) for h in data.hkl[audit_mask]}
    real_phase, real_refine, real_predict = phasing.solve_p1, autonomy.refine, autonomy.calculated_intensities
    calls = {"phase": 0, "refine": 0, "audit": 0}

    def phase_spy(cell, observations, *args, **kwargs):
        assert not any(tuple(h) in audit_indices for h in observations.hkl)
        calls["phase"] += 1
        return real_phase(cell, observations, *args, **kwargs)

    def refine_spy(model, observations, *args, **kwargs):
        assert not any(tuple(h) in audit_indices for h in observations.hkl)
        calls["refine"] += 1
        return real_refine(model, observations, *args, **kwargs)

    def audit_spy(model, hkl, *args, **kwargs):
        assert np.array_equal(hkl, data.hkl[audit_mask])
        calls["audit"] += 1
        return real_predict(model, hkl, *args, **kwargs)

    monkeypatch.setattr(phasing, "solve_p1", phase_spy)
    monkeypatch.setattr(autonomy, "refine", refine_spy)
    monkeypatch.setattr(autonomy, "calculated_intensities", audit_spy)
    arguments = dict(phasing_options={"algorithm": "hybrid", "trials": 2, "iterations": 400, "keep": 2},
                     refinement_options={"max_nfev": 120})
    first = autonomy.solve_refine(truth.cell, data, ["C"] * 4, **arguments)
    assert first["status"] == "completed"
    assert first["audit_evaluations"] == 1 and calls["audit"] == 1
    assert calls["phase"] == 1 and calls["refine"] == 3
    assert case.match_rms(truth, Structure.from_dict(first["refinement"]["structure"]))["rms_cartesian_angstrom"] < .005
    assert .8 < first["audit"]["sigma_rmse"] < 1.2
    altered = copy.deepcopy(data)
    altered.intensity[audit_mask] += 1e8
    altered.sigma[audit_mask] = 1e-6
    second = autonomy.solve_refine(truth.cell, altered, ["C"] * 4, **arguments)
    # Audit changes cannot alter candidates, their selection or the selected
    # refit. Its own diagnostic is intentionally allowed to change.
    assert first["candidates"] == second["candidates"]
    assert first["selected_candidate"] == second["selected_candidate"]
    assert first["refinement"] == second["refinement"]
    assert first["phasing"] == second["phasing"]
    assert second["audit"]["sigma_rmse"] > 1e10
    assert calls["audit"] == 2
    assert data.to_dict() == before
