"""Exact diffraction derivatives with bounded reflection work arrays.

The calculation differentiates the *physical* structure factor before applying
the coordinate, positive-ADP, occupancy and twin parameterizations.  It never
forms a normal matrix to estimate rank.  Reflections are evaluated in blocks;
the dense real intensity Jacobian required by SciPy remains N by P. The NumPy
reference uses a ``block_size`` by P complex temporary; the optional C++ path
contracts directly into a real block and retains only one reflection's site
contributions and compensated derivative sums.

``DiffractionPredictor`` is the shared interface for ordinary reflection least
squares and compressed overlapping-observation objectives.  Its parameterizer
is the native ``refinement._Parameterization`` object.  Site multiplicities are
the fixed Wyckoff-orbit branch of that parameterizer, not a model-selection
variable that changes when a trial coordinate happens to meet a special site.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.linalg import qr
from scipy.special import softmax

from .scattering import _indices, atomic_form_factor, validate_twin_laws
from .symmetry import SITE_TOLERANCE, operation_parts
from .kernels import cpp_intensity_block, pack_sites, requested_backend, resolve_backend


@dataclass
class AtomDifferential:
    """Nonzero physical tangent columns for one asymmetric-unit atom."""

    columns: np.ndarray
    xyz: np.ndarray                 # K x 3, fractional coordinates
    tensor: np.ndarray              # K x 3 x 3, Cartesian angstrom squared
    occupancy: np.ndarray           # K


@dataclass
class ParameterDifferentials:
    atoms: list[AtomDifferential]
    scale: np.ndarray
    fractions: np.ndarray


def symmetric_exponential_differential(matrix: Any, directions: Any) -> np.ndarray:
    """Fréchet derivative of exp(A) for real symmetric A, including degeneracy.

    In A's eigenbasis, ``D exp(A)[E] = K * E`` with
    ``K_ij = (exp(a_i)-exp(a_j))/(a_i-a_j)``.  The continuous repeated-eigenvalue
    limit is evaluated using sinh(x)/x.  Differentiating eigenvectors instead
    would be undefined at repeated ADP eigenvalues, such as an isotropic atom.
    """
    matrix = np.asarray(matrix, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if matrix.shape != (3, 3) or directions.ndim != 3 or directions.shape[1:] != (3, 3):
        raise ValueError("matrix exponential directions must have shape K x 3 x 3")
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    half_difference = (eigenvalues[:, None] - eigenvalues[None, :]) / 2
    ratio = np.ones_like(half_difference)
    np.divide(np.sinh(half_difference), half_difference, out=ratio,
              where=np.abs(half_difference) > 1e-12)
    divided_difference = np.exp((eigenvalues[:, None] + eigenvalues[None, :]) / 2) * ratio
    rotated = np.einsum("ia,kij,jb->kab", eigenvectors, directions, eigenvectors,
                        optimize=True)
    return np.einsum("ia,kab,jb->kij", eigenvectors,
                     divided_difference[None, :, :] * rotated, eigenvectors,
                     optimize=True)


def parameter_differentials(parameterization, values, model=None,
                            scale=None, fractions=None) -> ParameterDifferentials:
    """Exact sparse physical chain rules for every native parameter block."""
    values = np.asarray(values, dtype=float)
    if model is None or scale is None or fractions is None:
        model, scale, fractions = parameterization.unpack(values)
    npar = len(values)
    by_atom: list[dict[int, tuple[np.ndarray, np.ndarray, float]]] = [
        {} for _ in model.atoms
    ]
    dscale = np.zeros(npar)
    dfractions = np.zeros((len(fractions), npar))

    def add(label, column, xyz=None, tensor=None, occupancy=0.0):
        target = by_atom[parameterization.index[label]]
        if column not in target:
            target[column] = (np.zeros(3), np.zeros((3, 3)), 0.0)
        dx, du, do = target[column]
        if xyz is not None:
            dx = dx + xyz
        if tensor is not None:
            du = du + tensor
        target[column] = dx, du, do + occupancy

    for block in parameterization.blocks:
        kind, sl = block[:2]
        data = values[sl]
        columns = range(sl.start, sl.stop)
        if kind == "scale":
            dscale[sl] = scale
        elif kind == "twins":
            dfractions[:, sl] = (
                np.diag(fractions) - np.outer(fractions, fractions)
            )[:, 1:]
        elif kind == "xyz":
            tangent = block[5]
            for label in block[2]:
                for local, column in enumerate(columns):
                    add(label, column, xyz=tangent[:, local])
        elif kind in {"uiso", "chol", "log_tensor"}:
            if kind == "uiso":
                du = np.eye(3)[None, :, :] * np.exp(data[0])
            elif kind == "chol":
                lower = np.diag(np.exp(data[:3]))
                lower[1, 0], lower[2, 0], lower[2, 1] = data[3:]
                dlower = np.zeros((6, 3, 3))
                for axis in range(3):
                    dlower[axis, axis, axis] = lower[axis, axis]
                dlower[3, 1, 0] = dlower[4, 2, 0] = dlower[5, 2, 1] = 1.0
                du = dlower @ lower.T + lower @ dlower.transpose(0, 2, 1)
            else:
                log_tensor = np.einsum("k,kij->ij", data, block[3])
                du = symmetric_exponential_differential(log_tensor, block[3])
            for label in block[2]:
                for local, column in enumerate(columns):
                    add(label, column, tensor=du[local])
        elif kind == "occupancy":
            for label in block[2]:
                add(label, sl.start, occupancy=1.0)
        elif kind == "occupancy_sum":
            relative = softmax(np.r_[0.0, data])
            dweights = block[3] * (
                np.diag(relative) - np.outer(relative, relative)
            )[:, 1:]
            for i, label in enumerate(block[2]):
                for local, column in enumerate(columns):
                    add(label, column, occupancy=float(dweights[i, local]))
        else:
            raise ValueError(f"Unsupported analytic parameter block: {kind}")
    atoms = []
    for tangent in by_atom:
        columns = np.asarray(sorted(tangent), dtype=int)
        atoms.append(AtomDifferential(
            columns,
            np.asarray([tangent[j][0] for j in columns]).reshape(-1, 3),
            np.asarray([tangent[j][1] for j in columns]).reshape(-1, 3, 3),
            np.asarray([tangent[j][2] for j in columns]),
        ))
    return ParameterDifferentials(atoms, dscale, dfractions)


def physical_parameter_jacobian(parameterization, values):
    """Jacobian in the exact order returned by ``physical_values``."""
    model, scale, fractions = parameterization.unpack(values)
    derivatives = parameter_differentials(parameterization, values, model, scale, fractions)
    rows = [derivatives.scale]
    npar = len(values)
    for atom, tangent in zip(model.atoms, derivatives.atoms):
        xyz = np.zeros((3, npar))
        xyz[:, tangent.columns] = tangent.xyz.T
        rows.extend(xyz)
        occupancy = np.zeros(npar)
        occupancy[tangent.columns] = tangent.occupancy
        rows.append(occupancy)
        if atom.u_aniso is None:
            uiso = np.zeros(npar)
            uiso[tangent.columns] = np.trace(tangent.tensor, axis1=1, axis2=2) / 3
            rows.append(uiso)
        else:
            for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
                row = np.zeros(npar)
                row[tangent.columns] = tangent.tensor[:, i, j]
                rows.append(row)
    rows.extend(derivatives.fractions)
    return np.asarray(rows)


def _site_orbits(structure):
    """Cache coset representatives and the special-site tensor projector."""
    matrix = structure.cell.matrix
    inverse = np.linalg.inv(matrix)
    operations = []
    for op in structure.operations:
        rotation, translation = operation_parts(op)
        operations.append((rotation, translation, matrix @ rotation @ inverse))
    all_orbits = []
    for atom in structure.atoms:
        positions, groups = [], []
        for rotation, translation, rotation_cart in operations:
            position = (rotation @ atom.frac + translation) % 1
            for i, old in enumerate(positions):
                delta = position - old
                if np.max(np.abs(delta - np.rint(delta))) <= SITE_TOLERANCE:
                    groups[i][2].append(rotation_cart)
                    break
            else:
                positions.append(position)
                groups.append([rotation, translation, [rotation_cart]])
        all_orbits.append([(r, t, np.asarray(c)) for r, t, c in groups])
    return all_orbits


def _prepare_sites(model, orbits, tangents=None):
    prepared = []
    for i, (atom, orbit) in enumerate(zip(model.atoms, orbits)):
        u = np.eye(3) * atom.u_iso if atom.u_aniso is None else atom.u_aniso
        tangent = None if tangents is None else tangents[i]
        for rotation, translation, cart_rotations in orbit:
            position = (rotation @ atom.frac + translation) % 1
            if len(cart_rotations) == 1:
                cart = cart_rotations[0]
                rotated_u = cart @ u @ cart.T
            else:
                rotated_u = np.mean(cart_rotations @ u @ cart_rotations.transpose(0, 2, 1), axis=0)
            if tangent is not None and len(tangent.columns):
                dx = tangent.xyz @ rotation.T
                # 小张量直接批量相乘，避免每个原子重复规划 einsum 路径。
                if len(cart_rotations) == 1:
                    du = cart @ tangent.tensor @ cart.T
                else:
                    cart = cart_rotations[:, None, :, :]
                    du = np.mean(cart @ tangent.tensor @ cart.swapaxes(-1, -2), axis=0)
                prepared.append((i, position, rotated_u, tangent.columns,
                                 dx, du, tangent.occupancy))
            else:
                prepared.append((i, position, rotated_u, None, None, None, None))
    return prepared


class _StructureFactorPlan:
    def __init__(self, structure, hkl):
        self.hkl, _ = _indices(hkl)
        self.q = self.hkl @ structure.cell.reciprocal.T
        self.s2 = np.einsum("ij,ij->i", self.q, self.q) / 4
        elements = sorted({atom.element for atom in structure.atoms})
        self.element_indices = {element: i for i, element in enumerate(elements)}
        self.form_table = (np.asarray([atomic_form_factor(element, self.s2) for element in elements])
                           if elements else np.empty((0, len(self.hkl))))
        self.form_factors = {element: self.form_table[i] for i, element in enumerate(elements)}

    def block(self, model, sites, start, stop, npar, jacobian):
        hkl, q = self.hkl[start:stop], self.q[start:stop]
        f = np.zeros(stop - start, dtype=complex)
        df = np.zeros((stop - start, npar), dtype=complex) if jacobian else None
        for atom_index, position, u, columns, dx, du, docc in sites:
            atom = model.atoms[atom_index]
            factor = self.form_factors[atom.element][start:stop] + atom.f_prime + 1j * atom.f_double_prime
            exponent = -2 * np.pi**2 * np.einsum("ni,ij,nj->n", q, u, q)
            phase = 2j * np.pi * (hkl @ position)
            contribution = factor * np.exp(exponent + phase)
            f += atom.occupancy * contribution
            if jacobian and columns is not None:
                dphase = 2j * np.pi * (hkl @ dx.T)
                ddecay = -2 * np.pi**2 * np.einsum("ni,kij,nj->nk", q, du, q,
                                                  optimize=True)
                df[:, columns] += contribution[:, None] * (
                    docc[None, :] + atom.occupancy * (dphase + ddecay)
                )
        return f, df

    def intensity_block(self, model, sites, start, stop, npar, jacobian, packed=None):
        if packed is not None:
            intensity, derivative = cpp_intensity_block(
                self.hkl[start:stop], self.q[start:stop], self.form_table[:, start:stop],
                packed, npar if jacobian else 0)
            return intensity, derivative if jacobian else None
        f, df = self.block(model, sites, start, stop, npar, jacobian)
        return np.abs(f)**2, (2*np.real(np.conjugate(f)[:, None]*df) if jacobian else None)


class DiffractionPredictor:
    """Reusable intensity prediction and exact native-parameter Jacobian.

    ``twin_laws`` excludes the identity and acts on reciprocal column vectors.
    The underlying structure is the parameterizer's fixed cell, atom list,
    element types and Wyckoff-orbit topology.  Do not mutate the parameterizer
    after constructing a predictor.  Return arrays are cached: callers must
    not modify them in place.

    ``evaluate(values, jacobian=True)`` returns ``(model, intensity, J)``.
    ``backend`` chooses ``auto``, ``python`` or ``cpp``; if omitted it follows
    ``ARIADNE_NUMERIC_BACKEND`` (default auto). The NumPy reference's complex
    derivative work array uses at most B x P values. C++ computes a real block
    directly, with O(number of sites + P) scalar scratch storage. The returned
    real J necessarily uses N x P storage; ``jacobian=False`` avoids it entirely.
    """

    def __init__(self, parameterization, hkl, twin_laws=None, block_size=1024, backend=None):
        if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size < 1:
            raise ValueError("reflection block size must be a positive integer")
        self.parameterization = parameterization
        self.block_size = int(block_size)
        self.numeric_backend_requested = requested_backend(backend)
        self.numeric_backend = resolve_backend(self.numeric_backend_requested)
        indices, _ = _indices(hkl)
        self.hkl = indices
        laws, _ = validate_twin_laws(twin_laws, parameterization.fractions,
                                     parameterization.base.cell)
        self.plans = [_StructureFactorPlan(parameterization.base, indices @ law.T) for law in laws]
        self.orbits = _site_orbits(parameterization.base)
        self._last_values = None
        self._last_result = None
        self.evaluations = 0
        self.jacobian_evaluations = 0

    def validate_site_multiplicities(self, model):
        """Reject a final model that has crossed into another Wyckoff branch.

        Multiplicity is a discrete model choice. A continuous least-squares
        step must not silently halve an atom's scattering power when two
        symmetry images meet. Restart that case with explicit special-site
        coordinates and physically appropriate occupancies instead.
        """
        actual = _site_orbits(model)
        for atom, old, new in zip(model.atoms, self.orbits, actual):
            if len(old) != len(new):
                raise ValueError(
                    f"{atom.label} changed site multiplicity {len(old)} -> {len(new)}; "
                    "declare the new special-position model and restart refinement"
                )

    def evaluate(self, values, jacobian=True):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.parameterization.names),) or not np.isfinite(values).all():
            raise ValueError("native parameter vector has invalid shape or nonfinite values")
        if (self._last_values is not None and np.array_equal(values, self._last_values)
                and (not jacobian or self._last_result[2] is not None)):
            return (self._last_result if jacobian else
                    (self._last_result[0], self._last_result[1], None))
        model, scale, fractions = self.parameterization.unpack(values)
        npar, nobs = len(values), len(self.hkl)
        derivatives = (parameter_differentials(self.parameterization, values, model, scale, fractions)
                       if jacobian else None)
        sites = _prepare_sites(model, self.orbits, None if derivatives is None else derivatives.atoms)
        packed = (pack_sites(model, sites, self.plans[0].element_indices)
                  if self.numeric_backend == "cpp" else None)
        intensity = np.zeros(nobs)
        total_jacobian = np.zeros((nobs, npar)) if jacobian else None
        if jacobian:
            fraction_columns = [np.flatnonzero(direction) for direction in derivatives.fractions]
            scale_columns = np.flatnonzero(derivatives.scale)
        for start in range(0, nobs, self.block_size):
            stop = min(start + self.block_size, nobs)
            mixture = np.zeros(stop - start)
            if jacobian:
                # Accumulate directly in the final real output, rather than
                # retaining an additional B x P mixture and broadcast copies.
                dmixture = total_jacobian[start:stop]
            for domain, (plan, fraction) in enumerate(zip(self.plans, fractions)):
                if fraction == 0 and (not jacobian or not np.any(derivatives.fractions[domain])):
                    continue
                power, derivative = plan.intensity_block(model, sites, start, stop, npar, jacobian, packed)
                mixture += fraction * power
                if jacobian:
                    derivative *= fraction
                    dmixture += derivative
                    columns = fraction_columns[domain]
                    if len(columns):
                        dmixture[:, columns] += power[:, None] * derivatives.fractions[domain, columns]
            intensity[start:stop] = scale * mixture
            if jacobian:
                dmixture *= scale
                if len(scale_columns):
                    dmixture[:, scale_columns] += mixture[:, None] * derivatives.scale[scale_columns]
        if not np.isfinite(intensity).all() or (jacobian and not np.isfinite(total_jacobian).all()):
            raise ValueError("nonfinite native diffraction prediction or derivative")
        self.evaluations += 1
        self.jacobian_evaluations += int(jacobian)
        self._last_values = values.copy()
        self._last_result = model, intensity, total_jacobian
        return self._last_result


def right_singular_system(jacobian, block_size=1024):
    """Return singular values and a complete right basis without a tall U.

    Orthogonal triangular reduction obeys ``J = Q R`` and preserves singular
    values without squaring the condition number.  A streaming QR update
    accumulates only R and a row block.  SVD is then performed on the small R;
    full right vectors also expose the missing directions when N < P.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or not np.isfinite(jacobian).all():
        raise ValueError("Jacobian must be a finite matrix")
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError("QR block size must be a positive integer")
    nobs, npar = jacobian.shape
    if not npar:
        return np.empty(0), np.empty((0, 0))
    if not nobs:
        return np.empty(0), np.eye(npar)
    block_size = max(int(block_size), npar)
    triangular = np.empty((0, npar))
    for start in range(0, nobs, block_size):
        stacked = np.concatenate((triangular, jacobian[start:start + block_size]), axis=0)
        triangular = qr(stacked, mode="r", overwrite_a=True, check_finite=False)[0][:npar].copy()
    _, singular, vt = np.linalg.svd(triangular, full_matrices=True)
    return singular, vt
