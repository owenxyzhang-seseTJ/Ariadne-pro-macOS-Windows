"""Origin-aware symmetry proposals from a solved P1 density.

This is an independent implementation of the Fourier symmetry identity, not
the SHELXT program. Candidate groups are explicit. A small phase discrepancy
creates a model hypothesis; a reserved intensity set must choose between it
and competing hypotheses. P1 has zero discrepancy by definition.
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from .model import Structure
from .scattering import structure_factors
from .symmetry import get_operations, operation_parts, unique_sites


def _p1(model):
    if len(model.operations) != 1:
        raise ValueError("Supply an explicit full-unit-cell P1 model")
    if not 1 <= len(model.atoms) <= 512:
        raise ValueError("Symmetry proposal requires 1..512 unit-cell atoms")


def _distance(cell, left, right):
    delta = cell.minimum_image(left[:, None, :] - right[None, :, :])
    return np.linalg.norm(cell.to_cartesian(delta), axis=-1)


def phase_origin_search(model: Structure, hkl, spacegroup: str, *,
                        max_reflections=384, grid_size=4, keep_starts=6,
                        max_nfev=100):
    """Minimize |F(hR) exp(2πi[h.t+(hR-h).o])-F(h)|² over an origin o.

    The intensities of the candidate P1 model supply bounded phase weights.
    The returned discrepancy is a model-symmetry diagnostic, not a p-value or
    a probability for the space group. Caller supplies work-set indices only.
    Nonidentity translations test centering/screw/glide conditions as well.
    """
    _p1(model)
    for name, value, low, high in (("max_reflections", max_reflections, 8, 4096),
                                  ("grid_size", grid_size, 2, 8),
                                  ("keep_starts", keep_starts, 1, 32),
                                  ("max_nfev", max_nfev, 1, 1000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low},{high}]")
    h = np.asarray(hkl, dtype=float)
    if h.ndim != 2 or h.shape[1] != 3 or not len(h) or not np.isfinite(h).all() or not np.equal(h, np.rint(h)).all():
        raise ValueError("hkl must be a nonempty integral N by 3 matrix")
    h = np.unique(h.astype(int), axis=0)
    h = h[np.any(h != 0, axis=1)]
    if len(h) < 4:
        raise ValueError("At least four nonzero work indices are required")
    if len(h) > max_reflections:
        h = h[np.linspace(0, len(h)-1, max_reflections, dtype=int)]
    ops = get_operations(spacegroup, cell=model.cell)
    transforms = [(r, t) for r, t in map(operation_parts, ops)
                  if not (np.array_equal(r, np.eye(3)) and np.allclose(t % 1, 0))]
    if not transforms:
        return {"spacegroup": spacegroup, "origin_shift": [0., 0., 0.],
                "phase_discrepancy": 0., "origin_rank": 0, "n_reflections": len(h),
                "n_operations": len(ops), "evaluation_count": 0,
                "interpretation": "Identity symmetry has no discriminating phase test"}
    f = structure_factors(model, h)
    rhs = np.tile(f, len(transforms))
    q = np.concatenate([h @ r - h for r, _ in transforms])
    ht = np.concatenate([h @ t for _, t in transforms])
    transformed = np.concatenate([structure_factors(model, np.rint(h @ r).astype(int))
                                  for r, _ in transforms])
    normalizer = np.sqrt(max(np.sum(np.abs(rhs)**2 + np.abs(transformed)**2), 1e-30))
    def residual(origin):
        value = (transformed * np.exp(2j*np.pi*(ht + q @ origin)) - rhs) / normalizer
        return np.r_[value.real, value.imag]
    def jacobian(origin):
        d = (2j*np.pi/normalizer * transformed * np.exp(2j*np.pi*(ht + q @ origin)))[:, None] * q
        return np.concatenate([d.real, d.imag])
    starts = [np.array(v)/grid_size for v in itertools.product(range(grid_size), repeat=3)]
    # Atom-pair origins complement the bounded reciprocal-space grid. For an
    # inversion operation this includes the exact centre-to-origin solution.
    rotation, translation = max(transforms, key=lambda rt: np.linalg.matrix_rank(rt[0]-np.eye(3)))
    system = rotation - np.eye(3)
    inverse = np.linalg.pinv(system)
    ranked = sorted(model.atoms, key=lambda a: a.element)
    for a in ranked[:24]:
        for b in ranked[:24]:
            if a.element == b.element:
                delta = b.frac - rotation @ a.frac - translation
                starts.append((inverse @ delta) % 1)
    ranked_starts = sorted(starts, key=lambda x: float(residual(x) @ residual(x)))[:keep_starts]
    fits = [least_squares(residual, x, jac=jacobian, max_nfev=max_nfev,
                          ftol=1e-10, xtol=1e-10, gtol=1e-10) for x in ranked_starts]
    best = min(fits, key=lambda x: float(x.fun @ x.fun))
    return {"spacegroup": spacegroup, "origin_shift": (best.x % 1).tolist(),
            "phase_discrepancy": float(best.fun @ best.fun),
            "origin_rank": int(np.linalg.matrix_rank(q)), "n_reflections": len(h),
            "n_operations": len(ops), "evaluation_count": sum(x.nfev for x in fits),
            "numerical_convergence": bool(best.success),
            "interpretation": "Conditional Fourier symmetry discrepancy of the P1 model"}


def project_spacegroup(model: Structure, spacegroup: str, origin_shift, *, tolerance=0.18):
    """Convert a compatible full-cell P1 model to a symmetry-exact ASU model.

    One-to-one same-element matching prevents several atoms from claiming the
    same site. Occupancy is a physical site occupancy and is never divided by
    group order. An incompatible orbit or incomplete composition is rejected.
    """
    _p1(model)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Matching tolerance must be finite and positive, in angstrom")
    shift = np.asarray(origin_shift, dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("origin_shift must have three finite entries")
    shifted = copy.deepcopy(model)
    for atom in shifted.atoms:
        atom.frac = (atom.frac + shift) % 1
    xyz = np.array([a.frac for a in shifted.atoms])
    ops = get_operations(spacegroup, cell=model.cell)
    parts = list(map(operation_parts, ops))
    by_element = {}
    for i, atom in enumerate(shifted.atoms):
        by_element.setdefault(atom.element, []).append(i)
    parent = np.arange(len(xyz))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)
    mappings = []
    maximum = 0.
    for rotation, translation in parts:
        mapped = xyz @ rotation.T + translation
        assignment = np.empty(len(xyz), int)
        for ids in by_element.values():
            ids = np.array(ids)
            costs = _distance(model.cell, mapped[ids], xyz[ids])
            rows, cols = linear_sum_assignment(costs)
            maximum = max(maximum, float(costs[rows, cols].max()))
            if np.any(costs[rows, cols] > tolerance):
                raise ValueError(f"P1 atoms do not form {spacegroup} orbits within {tolerance:g} angstrom")
            assignment[ids[rows]] = ids[cols]
            for i, j in zip(ids[rows], ids[cols]):
                parent[root(j)] = root(i)
        mappings.append(assignment)
    orbits = {}
    for i in range(len(xyz)):
        orbits.setdefault(root(i), []).append(i)
    atoms = []
    for members in orbits.values():
        lead = members[0]
        estimates, tensors = [], []
        # The representation of an orbit must not depend on which member
        # happens to occur first in the input. Promote a mixed orbit, and
        # treat an isotropic member as its physical Cartesian tensor u*I.
        anisotropic_orbit = any(shifted.atoms[i].u_aniso is not None for i in members)
        for (rotation, translation), assignment in zip(parts, mappings):
            j = assignment[lead]
            predicted = rotation @ xyz[lead] + translation
            matched = predicted + model.cell.minimum_image(xyz[j] - predicted)
            estimates.append(np.linalg.solve(rotation, matched-translation))
            if anisotropic_orbit:
                other = shifted.atoms[j]
                u = other.u_aniso if other.u_aniso is not None else np.eye(3)*other.u_iso
                cart_rotation = model.cell.matrix @ rotation @ np.linalg.inv(model.cell.matrix)
                inverse = np.linalg.inv(cart_rotation)
                tensors.append(inverse @ u @ inverse.T)
        atom = copy.deepcopy(shifted.atoms[lead])
        occupancies = [shifted.atoms[i].occupancy for i in members]
        if np.ptp(occupancies) > 1e-5:
            raise ValueError("A proposed symmetry orbit has inconsistent physical occupancies")
        atom.frac = np.mean(estimates, axis=0) % 1
        atom.u_iso = float(np.mean([shifted.atoms[i].u_iso for i in members]))
        if tensors:
            atom.u_aniso = np.mean(tensors, axis=0)
            atom.u_iso = float(np.trace(atom.u_aniso)/3)
        atoms.append(atom)
    result = Structure(model.cell, atoms, spacegroup, model.wavelength)
    expanded = [(atom.element, pos) for atom in result.atoms for pos, _ in unique_sites(result, atom)]
    if len(expanded) != len(model.atoms):
        raise ValueError("Symmetry projection changed the number of physical unit-cell sites")
    squared = []
    for element, ids in by_element.items():
        positions = np.array([p for e, p in expanded if e == element])
        costs = _distance(model.cell, xyz[ids], positions)
        rows, cols = linear_sum_assignment(costs)
        squared.extend(costs[rows, cols]**2)
    rms = float(np.sqrt(np.mean(squared)))
    if rms > tolerance:
        raise ValueError("Symmetrized model exceeds coordinate projection tolerance")
    return result, {"rms_cartesian_angstrom": rms, "maximum_operation_mismatch": maximum,
                    "unit_cell_atoms": len(expanded), "asymmetric_unit_atoms": len(atoms),
                    "origin_shift": (shift % 1).tolist()}


def propose_spacegroups(model, hkl, candidates, *, tolerance=0.18, phase_limit=0.08):
    """Make auditable model hypotheses; no hardcoded preference for high symmetry."""
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 32:
        raise ValueError("Provide 1..32 explicit space-group candidates")
    if not np.isfinite(phase_limit) or phase_limit < 0:
        raise ValueError("phase_limit must be finite and nonnegative")
    result = []
    for name in dict.fromkeys(candidates):
        try:
            diagnostic = phase_origin_search(model, hkl, name)
            if diagnostic["phase_discrepancy"] > phase_limit:
                result.append({**diagnostic, "eligible": False, "reason": "phase discrepancy"})
                continue
            projected, geometry = project_spacegroup(model, name, diagnostic["origin_shift"], tolerance=tolerance)
            result.append({**diagnostic, "eligible": True, "projection": geometry, "structure": projected})
        except (ValueError, RuntimeError) as error:
            result.append({"spacegroup": name, "eligible": False, "reason": str(error)})
    return result
