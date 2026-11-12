"""Reproduce derivative accuracy, local-refinement timing and SVD memory checks.

Run from the native project after installing its ``dev`` dependencies::

    OPENBLAS_NUM_THREADS=1 python validation/run_v08_refinement.py

This benchmark measures a synthetic *local refinement*, not ab initio structure
solution or a comparison against SHELXL. The finite-difference comparison uses
the same cached forward model, residual definition, starting point and solver.
The default explicitly selects the Python/NumPy predictor, even if an optional
C++ extension is installed. C++ timing is measured in run_v08_kernels.py.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import subprocess
import sys
import time

import gemmi
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from ariadne_xtal.derivatives import DiffractionPredictor, right_singular_system
from ariadne_xtal.kernels import kernel_status
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.refinement import RefinementOptions, refine
from ariadne_xtal.scattering import calculated_intensities


HERE = Path(__file__).resolve().parent


def memory_worker(method):
    matrix = np.random.default_rng(442).normal(size=(3500, 64))
    if method == "full_svd":
        left, singular, right = np.linalg.svd(matrix, full_matrices=True)
        checksum = float(singular.sum() + left[0, 0] + right[0, 0])
    else:
        singular, right = right_singular_system(matrix, block_size=256)
        checksum = float(singular.sum() + right[0, 0])
    return {"method": method, "shape": list(matrix.shape),
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "checksum": checksum}


def derivatives():
    helpers = runpy.run_path(str(HERE.parent / "tests/test_refinement_v08.py"))
    result = []
    for name in ("oblique_anomalous", "special_repeated_eigenvalues", "nonsymmorphic",
                 "occupancy_and_inversion_twin", "merohedral_three_domains"):
        structure, options, constraints = helpers["_case"](name)
        parameterization, values, laws = helpers["_parameterization"](structure, options, constraints)
        hkl = helpers["_indices"]()
        _, intensity, analytic = DiffractionPredictor(parameterization, hkl, laws, block_size=17).evaluate(values)

        def reference(x):
            model, scale, fractions = parameterization.unpack(x)
            return calculated_intensities(model, hkl, scale, laws, fractions)

        numeric = np.empty_like(analytic)
        for column in range(len(values)):
            step = 2e-6*max(1., abs(values[column]))
            plus, minus = values.copy(), values.copy()
            plus[column] += step
            minus[column] -= step
            numeric[:, column] = (reference(plus)-reference(minus))/(2*step)
        scaled_error = np.abs(analytic-numeric)/np.maximum(np.max(np.abs(numeric), axis=0), 1.)
        result.append({"case": name, "n_reflections": len(hkl), "n_parameters": len(values),
                       "maximum_scaled_derivative_error": float(np.max(scaled_error)),
                       "maximum_intensity_error": float(np.max(np.abs(intensity-reference(values))))})
    return result


def benchmark():
    rng = np.random.default_rng(419)
    truth = Structure(Cell(11, 12.5, 14.3, 81, 96, 104), [
        Atom(f"{element}{i+1}", element, rng.uniform(.08, .92, 3), u_iso=rng.uniform(.018, .04))
        for i, element in enumerate(["Zn", "Zn"] + ["C", "O", "N", "C"]*4)
    ])
    hkl = np.asarray([h for h in itertools.product(range(-6, 7), repeat=3)
                      if h != (0, 0, 0) and tuple(h) < tuple(-x for x in h)], dtype=int)
    # Independent published-coefficient structure-factor calculator from Gemmi.
    gemmi_structure = gemmi.SmallStructure()
    gemmi_structure.cell = gemmi.UnitCell(**truth.cell.to_dict())
    gemmi_structure.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    gemmi_structure.setup_cell_images()
    for atom in truth.atoms:
        site = gemmi.SmallStructure.Site()
        site.label, site.element = atom.label, gemmi.Element(atom.element)
        site.fract, site.occ, site.u_iso = gemmi.Fractional(*atom.frac), atom.occupancy, atom.u_iso
        gemmi_structure.add_site(site)
    calculator = gemmi.StructureFactorCalculatorX(gemmi_structure.cell)
    ideal = 1.6*np.abs(np.asarray([
        calculator.calculate_sf_from_small_structure(gemmi_structure, h.tolist()) for h in hkl
    ]))**2
    sigma = np.sqrt(ideal + 100.)
    sigma[::11] = 500.
    observed = ideal + rng.normal(0, sigma)
    data = Reflections(hkl, observed, sigma)
    start = copy.deepcopy(truth)
    for atom in start.atoms[1:]:
        atom.frac += rng.normal(0, .002, 3)
        atom.u_iso *= 1.12
    measured = []
    last = {}
    # Warm numerical-library dispatch without warming one solver method only.
    calculated_intensities(start, hkl)
    for repeat in range(3):
        for method in ("analytic", "finite_difference"):
            wall, cpu = time.perf_counter(), time.process_time()
            result = refine(start, data, RefinementOptions(jacobian=method,
                                                           reflection_block_size=512))
            elapsed, cpu_elapsed = time.perf_counter()-wall, time.process_time()-cpu
            measured.append({"method": method, "repeat": repeat+1, "wall_seconds": elapsed,
                             "cpu_seconds": cpu_elapsed,
                             "numeric_backend": result.metrics["numeric_backend"],
                             "diffraction_evaluations": result.metrics["diffraction_evaluations"],
                             "objective_evaluations": result.metrics["objective_evaluations_total"],
                             "chi_squared": result.metrics["chi_squared"], "success": result.success,
                             "negative_intensities": result.metrics["negative_intensities"]})
            last[method] = result
    summary = {}
    for method in last:
        rows = [r for r in measured if r["method"] == method]
        summary[method] = {"median_wall_seconds": float(np.median([r["wall_seconds"] for r in rows])),
                           "median_cpu_seconds": float(np.median([r["cpu_seconds"] for r in rows])),
                           "diffraction_evaluations": rows[0]["diffraction_evaluations"]}
    delta = last["analytic"].calculated-last["finite_difference"].calculated
    return {"kind": "synthetic_local_refinement", "observation_generator": "Gemmi independent SF calculator",
            "n_atoms": len(truth.atoms), "n_reflections": len(hkl),
            "n_parameters": last["analytic"].metrics["n_parameters"],
            "median_wall_speed_ratio": summary["finite_difference"]["median_wall_seconds"]/summary["analytic"]["median_wall_seconds"],
            "prediction_difference_sigma_rms": float(np.sqrt(np.mean((delta/sigma)**2))),
            "summary": summary, "runs": measured,
            "interpretation": "Local timing in one environment, not a performance guarantee or SHELXL comparison"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["python", "cpp", "auto"], default="python")
    parser.add_argument("--memory-worker", choices=["full_svd", "streamed_qr"])
    parser.add_argument("--output", type=Path, default=HERE/"v08_refinement_measurements.json")
    args = parser.parse_args()
    os.environ["ARIADNE_NUMERIC_BACKEND"] = args.backend
    with threadpool_limits(limits=1):
        if args.memory_worker:
            print(json.dumps(memory_worker(args.memory_worker)))
            return
        result = {"schema": "ariadne_xtal.v08_refinement_validation/1",
                  "python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "gemmi": gemmi.__version__,
                  "numeric_backend": kernel_status(args.backend),
                  "blas_threads": 1, "derivatives": derivatives(), "refinement": benchmark()}
        result["source_sha256"] = {
            name: hashlib.sha256((HERE.parent/"src/ariadne_xtal"/name).read_bytes()).hexdigest()
            for name in ("derivatives.py", "refinement.py", "scattering.py", "kernels.py")
        }
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result["memory"] = [json.loads(subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--memory-worker", method], env=env, text=True))
            for method in ("full_svd", "streamed_qr")]
        result["memory_interpretation"] = (
            "Separate Linux processes; peak RSS includes Python and loaded libraries. Both use identical 3500 x 64 input. "
            "The old full left-singular matrix alone has 98,000,000 bytes; the new QR update stores no such matrix. "
            "Dense least_squares still retains the real N x P Jacobian."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output)
    print(json.dumps(result["refinement"]["summary"], indent=2))
    print(json.dumps(result["memory"], indent=2))


if __name__ == "__main__":
    main()
