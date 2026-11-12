"""Optional, observable native acceleration of the existing physical model.

``ARIADNE_NUMERIC_BACKEND=auto|python|cpp`` chooses the implementation.  Auto
uses the compiled extension when importable, otherwise the NumPy reference.
Explicit cpp fails clearly when unavailable; numerical errors never trigger
a silent fallback.  The C++ kernel is scalar, releases the GIL, and starts no
OpenMP or other worker threads.  Outer agent scheduling owns concurrency.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral
import os

import numpy as np

try:
    from . import _intensity_cpp as _cpp
except ImportError as error:
    _cpp = None
    _import_error = str(error)
else:
    _import_error = None


def requested_backend(value=None):
    value = os.environ.get("ARIADNE_NUMERIC_BACKEND", "auto") if value is None else value
    if not isinstance(value, str) or value not in {"auto", "python", "cpp"}:
        raise ValueError("Numeric backend must be auto, python or cpp")
    return value


def resolve_backend(value=None):
    choice = requested_backend(value)
    if choice == "cpp" and _cpp is None:
        raise RuntimeError("C++ numeric backend is unavailable; build with "
                           "python tools/build_optional_kernel.py --inplace or select python/auto")
    return "cpp" if choice != "python" and _cpp is not None else "python"


def kernel_status(value=None):
    """Describe both availability and the requested, actually usable backend."""
    choice = requested_backend(value)
    available = _cpp is not None
    return {
        "requested": choice,
        "resolved": ("cpp" if available else "unavailable") if choice == "cpp" else
                    ("cpp" if choice == "auto" and available else "python"),
        "compiled_available": available,
        "build": None if not available else _cpp.build_info(),
        "fallback_reason": _import_error if choice == "auto" and not available else None,
        "thread_policy": "C++ kernel uses one thread and releases GIL; agent workers own concurrency",
        "scope": "Structure-factor intensity and analytic derivative accumulation; no statistical-model substitution",
    }


@dataclass(frozen=True)
class PackedSites:
    """Per-evaluation site geometry and sparse physical tangent columns."""

    data: np.ndarray             # S x 12: xyz, Uxx/Uyy/Uzz/Uxy/Uxz/Uyz, occ, f', f''
    element_indices: np.ndarray  # S
    offsets: np.ndarray          # S+1 CSR pointers
    columns: np.ndarray          # T global parameter indices
    tangents: np.ndarray         # T x 10: dxyz, symmetric dU, docc


def _tensor_six(tensor):
    tensor = np.asarray(tensor)
    return np.stack((tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2],
                     (tensor[..., 0, 1]+tensor[..., 1, 0])/2,
                     (tensor[..., 0, 2]+tensor[..., 2, 0])/2,
                     (tensor[..., 1, 2]+tensor[..., 2, 1])/2), axis=-1)


def pack_sites(model, sites, element_indices):
    """Pack only site-sized arrays, independently of the reflection count."""
    count = len(sites)
    tangent_count = sum(0 if site[3] is None else len(site[3]) for site in sites)
    data = np.empty((count, 12), dtype=np.float64)
    elements = np.empty(count, dtype=np.int64)
    offsets = np.empty(count+1, dtype=np.int64)
    columns = np.empty(tangent_count, dtype=np.int64)
    directions = np.empty((tangent_count, 10), dtype=np.float64)
    cursor = 0
    for i, (atom_index, position, tensor, indices, dx, du, docc) in enumerate(sites):
        atom = model.atoms[atom_index]
        data[i, :3] = position
        data[i, 3:9] = _tensor_six(tensor)
        data[i, 9:] = atom.occupancy, atom.f_prime, atom.f_double_prime
        elements[i] = element_indices[atom.element]
        offsets[i] = cursor
        if indices is not None:
            end = cursor+len(indices)
            columns[cursor:end] = indices
            directions[cursor:end, :3] = dx
            directions[cursor:end, 3:9] = _tensor_six(du)
            directions[cursor:end, 9] = docc
            cursor = end
    offsets[-1] = cursor
    return PackedSites(data, elements, offsets, columns, directions)


def cpp_intensity_block(hkl, q, form_factors, sites, n_parameters):
    """Evaluate one block with checked, aligned native arrays.

    Inputs are not mutated. Non-contiguous/unaligned numerical views are
    copied to suitable block buffers. Callers must not mutate shared inputs
    while the GIL-releasing calculation is in progress. C++ uses separate
    double real/imaginary scalars internally, so complex128 ABI assumptions
    are absent from the binding.
    """
    if _cpp is None:
        resolve_backend("cpp")
    if (isinstance(n_parameters, bool) or not isinstance(n_parameters, Integral)
            or n_parameters < 0):
        raise ValueError("n_parameters must be a nonnegative integer")
    raw_hkl = np.asarray(hkl)
    if raw_hkl.dtype.kind not in {"i", "u"}:
        raise TypeError("Kernel Miller indices must be integers")
    if raw_hkl.size and (np.any(raw_hkl > np.iinfo(np.int32).max)
                          or np.any(raw_hkl < np.iinfo(np.int32).min)):
        raise ValueError("Kernel Miller indices exceed supported int32 range")
    required = lambda value, dtype: np.require(value, dtype=dtype, requirements=["C", "A"])
    indices = required(raw_hkl, np.int64)
    reciprocal = required(q, np.float64)
    factors = required(form_factors, np.float64)
    data = required(sites.data, np.float64)
    elements = required(sites.element_indices, np.int64)
    offsets = required(sites.offsets, np.int64)
    columns = required(sites.columns, np.int64)
    tangents = required(sites.tangents, np.float64)
    intensity = np.empty(len(indices), dtype=np.float64)
    jacobian = np.empty((len(indices), n_parameters), dtype=np.float64)
    _cpp.intensity_block(indices, reciprocal, factors, data, elements, offsets,
                         columns, tangents, intensity, jacobian)
    return intensity, jacobian


if __name__ == "__main__":
    print(json.dumps(kernel_status(), indent=2, allow_nan=False))
