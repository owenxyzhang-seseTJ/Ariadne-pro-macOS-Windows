"""Reproduce optional C++ kernel timing, memory and numerical checks.

From the native source tree with the dev extras installed::

    python tools/build_optional_kernel.py --inplace
    OPENBLAS_NUM_THREADS=1 python validation/run_v08_kernels.py

All comparisons use the same physical model and analytic derivative. This
script does not compare finite differences, SHELXL, or ab initio success rates.
The peak-RSS comparisons run in separate Linux processes with identical input.
"""
from __future__ import annotations

import argparse
import copy
import cProfile
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import subprocess
import sys
import time

import gemmi
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from ariadne_xtal.derivatives import DiffractionPredictor
from ariadne_xtal.kernels import kernel_status, _cpp
from ariadne_xtal.model import Atom, Cell, Reflections, Structure
from ariadne_xtal.refinement import RefinementOptions, _Parameterization, refine


HERE = Path(__file__).resolve().parent


def problem(name):
    rng = np.random.default_rng(882)
    if name == "three_atoms_three_domains":
        model = Structure(Cell(8.1, 8.1, 10.3), [
            Atom("Zn1", "Zn", [.123, .273, .319], f_prime=-.6, f_double_prime=2.3,
                 u_aniso=np.array([[.023, .002, -.003], [.002, .031, .004], [-.003, .004, .027]])),
            Atom("O1", "O", [.329, .417, .531], u_iso=.036),
            Atom("C1", "C", [.253, .571, .387], u_iso=.024),
        ])
        laws = [np.diag([-1, -1, 1]), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])]
        fractions = np.array([.55, .3, .15])
        options = RefinementOptions(twin_laws=[law.tolist() for law in laws],
                                    twin_fractions=fractions.tolist(), refine_twin_fractions=True)
        hkl = np.tile(np.array([h for h in itertools.product(range(-4, 5), repeat=3)
                               if h != (0, 0, 0)], dtype=int), (10, 1))
    else:
        count = 32 if name == "thirty_two_atoms" else 80
        nobs = 4096 if count == 32 else 8192
        model = Structure(Cell(13., 15., 17., 82., 96., 104.), [
            Atom(f"{element}{i+1}", element, rng.uniform(.05, .95, 3),
                 u_iso=rng.uniform(.015, .04))
            for i, element in enumerate((["Zn", "O", "N", "C"]*(count//4)))
        ])
        pool = np.array([h for h in itertools.product(range(-11, 12), repeat=3)
                         if h != (0, 0, 0)], dtype=int)
        hkl = pool[rng.choice(len(pool), nobs, replace=False)]
        laws, fractions, options = [], np.ones(1), RefinementOptions()
    parameterization = _Parameterization(model, options, 1.7, fractions, [])
    return parameterization, np.asarray(parameterization.start), hkl, laws


def sequence(values, count):
    answer = []
    for i in range(count):
        candidate = values.copy()
        candidate[0] += .0001*(i+1)
        candidate[-1] += .0002*(i+1)
        answer.append(candidate)
    return answer


def predictor_benchmark(name, repeats=5):
    parameters, values, hkl, laws = problem(name)
    choices = sequence(values, 8)
    predictors = {backend: DiffractionPredictor(parameters, hkl, laws, block_size=1024,
                                                backend=backend)
                  for backend in ("python", "cpp")}
    for predictor in predictors.values():
        predictor.evaluate(values)
    rows = []
    for repeat in range(repeats):
        for backend in (("python", "cpp") if repeat % 2 == 0 else ("cpp", "python")):
            cpu, wall = time.process_time(), time.perf_counter()
            for candidate in choices:
                predictors[backend].evaluate(candidate)
            elapsed, cpu_elapsed = time.perf_counter()-wall, time.process_time()-cpu
            rows.append({"backend": backend, "repeat": repeat+1,
                         "wall_seconds": elapsed, "cpu_seconds": cpu_elapsed})
    summary = {backend: {"median_wall_seconds": float(np.median([
        row["wall_seconds"] for row in rows if row["backend"] == backend])),
        "median_cpu_seconds": float(np.median([
            row["cpu_seconds"] for row in rows if row["backend"] == backend]))}
        for backend in predictors}
    reference = predictors["python"].evaluate(choices[-1])
    actual = predictors["cpp"].evaluate(choices[-1])
    column_scale = np.maximum(np.max(np.abs(reference[2]), axis=0), 1.)
    return {"case": name, "n_atoms": len(parameters.base.atoms), "n_domains": len(laws)+1,
            "n_reflections": len(hkl), "n_parameters": len(values), "block_size": 1024,
            "evaluations_per_run": len(choices), "summary": summary, "runs": rows,
            "median_wall_speedup": summary["python"]["median_wall_seconds"]/summary["cpp"]["median_wall_seconds"],
            "maximum_intensity_absolute_difference": float(np.max(np.abs(reference[1]-actual[1]))),
            "maximum_scaled_jacobian_difference": float(np.max(np.abs(reference[2]-actual[2])/column_scale)),
            "timing_scope": "Cached predictor evaluate, including parameter unpacking, symmetry-site preparation, all domains, intensity and analytic Jacobian; fresh values defeat result cache"}


def profile_reference():
    parameters, values, hkl, laws = problem("three_atoms_three_domains")
    predictor = DiffractionPredictor(parameters, hkl, laws, block_size=1024, backend="python")
    candidates = sequence(values, 12)
    profiler = cProfile.Profile()
    profiler.enable()
    for candidate in candidates:
        predictor.evaluate(candidate)
    profiler.disable()
    stats = pstats.Stats(profiler)
    functions = []
    block_time = 0.
    for (filename, line, name), (primitive_calls, calls, own, cumulative, _) in stats.stats.items():
        if name == "block" and filename.endswith("derivatives.py"):
            block_time += cumulative
        functions.append({"file": Path(filename).name, "line": line, "function": name,
                          "calls": calls, "primitive_calls": primitive_calls,
                          "own_seconds": own, "cumulative_seconds": cumulative})
    return {"backend": "python", "evaluations": len(candidates),
            "case": "three_atoms_three_domains", "total_profiled_seconds": stats.total_tt,
            "structure_factor_block_cumulative_seconds": block_time,
            "structure_factor_block_fraction": block_time/stats.total_tt,
            "top_cumulative_functions": sorted(functions, key=lambda row: row["cumulative_seconds"], reverse=True)[:15],
            "interpretation": "cProfile selects the hotspot; its instrumented times are not used as the reported speedup"}


def independent_intensities(model, hkl):
    small = gemmi.SmallStructure()
    small.cell = gemmi.UnitCell(**model.cell.to_dict())
    small.spacegroup = gemmi.find_spacegroup_by_name(model.spacegroup)
    small.setup_cell_images()
    for atom in model.atoms:
        site = gemmi.SmallStructure.Site()
        site.label, site.element = atom.label, gemmi.Element(atom.element)
        site.fract, site.occ, site.u_iso = gemmi.Fractional(*atom.frac), atom.occupancy, atom.u_iso
        small.add_site(site)
    calculator = gemmi.StructureFactorCalculatorX(small.cell)
    return np.abs(np.asarray([calculator.calculate_sf_from_small_structure(small, h.tolist())
                              for h in hkl]))**2


def refinement_benchmark():
    rng = np.random.default_rng(419)
    truth = Structure(Cell(11, 12.5, 14.3, 81, 96, 104), [
        Atom(f"{element}{i+1}", element, rng.uniform(.08, .92, 3), u_iso=rng.uniform(.018, .04))
        for i, element in enumerate(["Zn", "Zn"] + ["C", "O", "N", "C"]*4)
    ])
    hkl = np.asarray([h for h in itertools.product(range(-6, 7), repeat=3)
                      if h != (0, 0, 0) and tuple(h) < tuple(-x for x in h)], dtype=int)
    ideal = 1.6*independent_intensities(truth, hkl)
    sigma = np.sqrt(ideal+100.)
    sigma[::11] = 500.
    observations = Reflections(hkl, ideal+rng.normal(0, sigma), sigma)
    start = copy.deepcopy(truth)
    for atom in start.atoms[1:]:
        atom.frac += rng.normal(0, .002, 3)
        atom.u_iso *= 1.12
    rows, last = [], {}
    previous = os.environ.get("ARIADNE_NUMERIC_BACKEND")
    try:
        for repeat in range(3):
            for backend in (("python", "cpp") if repeat % 2 == 0 else ("cpp", "python")):
                os.environ["ARIADNE_NUMERIC_BACKEND"] = backend
                cpu, wall = time.process_time(), time.perf_counter()
                fitted = refine(start, observations, RefinementOptions(jacobian="analytic", reflection_block_size=512))
                elapsed, cpu_elapsed = time.perf_counter()-wall, time.process_time()-cpu
                rows.append({"backend": backend, "actual_backend": fitted.metrics["numeric_backend"],
                             "repeat": repeat+1, "wall_seconds": elapsed, "cpu_seconds": cpu_elapsed,
                             "success": fitted.success, "chi_squared": fitted.metrics["chi_squared"],
                             "diffraction_evaluations": fitted.metrics["diffraction_evaluations"],
                             "jacobian_rank": fitted.metrics["jacobian_rank"],
                             "negative_intensities": fitted.metrics["negative_intensities"]})
                last[backend] = fitted
    finally:
        if previous is None:
            os.environ.pop("ARIADNE_NUMERIC_BACKEND", None)
        else:
            os.environ["ARIADNE_NUMERIC_BACKEND"] = previous
    summary = {backend: {"median_wall_seconds": float(np.median([
        row["wall_seconds"] for row in rows if row["backend"] == backend])),
        "median_cpu_seconds": float(np.median([
            row["cpu_seconds"] for row in rows if row["backend"] == backend]))}
        for backend in last}
    delta = last["cpp"].calculated-last["python"].calculated
    return {"kind": "synthetic_local_refinement", "observation_generator": "Gemmi independent SF calculator",
            "n_atoms": len(truth.atoms), "n_reflections": len(hkl),
            "n_parameters": last["python"].metrics["n_parameters"], "summary": summary, "runs": rows,
            "median_wall_speedup": summary["python"]["median_wall_seconds"]/summary["cpp"]["median_wall_seconds"],
            "prediction_difference_sigma_rms": float(np.sqrt(np.mean((delta/sigma)**2))),
            "chi_squared_difference": last["cpp"].metrics["chi_squared"]-last["python"].metrics["chi_squared"],
            "timing_scope": "Entire refine call, including initialization, analytic least squares, restraints, rank and covariance; same starting model, no ab initio or external-software comparison"}


def memory_worker(backend, block_size):
    parameters, values, hkl, laws = problem("eighty_atoms_memory")
    predictor = DiffractionPredictor(parameters, hkl, laws, block_size=block_size, backend=backend)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model, intensity, jacobian = predictor.evaluate(values)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"backend": backend, "block_size": block_size, "n_atoms": len(model.atoms),
            "n_reflections": len(hkl), "n_parameters": len(values),
            "baseline_peak_rss_kib": before, "peak_process_rss_kib": after,
            "increase_over_baseline_peak_kib": after-before,
            "returned_jacobian_bytes": jacobian.nbytes,
            "numpy_complex_derivative_block_bytes": 16*min(block_size, len(hkl))*len(values),
            "checksum": float(intensity.sum()+jacobian.sum())}


def cancellation_check():
    amplitudes = [1e16, 1., -1e16, 1.]
    with localcontext() as context:
        context.prec = 80
        total = sum(Decimal.from_float(value) for value in amplitudes)
        expected = float(total*total)
    sites = np.zeros((4, 12)); sites[:, 9] = 1.; sites[:, 10] = amplitudes
    tangents = np.zeros((4, 10)); tangents[:, 9] = 1.
    power, jacobian = np.empty(1), np.empty((1, 1))
    _cpp.intensity_block(np.zeros((1, 3), np.int64), np.zeros((1, 3)), np.zeros((1, 1)),
                         sites, np.zeros(4, np.int64), np.arange(5, dtype=np.int64),
                         np.zeros(4, np.int64), tangents, power, jacobian)
    model = Structure(Cell(8.3, 9.1, 11.2), [
        Atom("Zn1", "Zn", [.123, .267, .319], u_iso=.027),
        Atom("O1", "O", [.357, .421, .537], u_iso=.031),
    ], "P 21 21 21")
    parameters = _Parameterization(model, RefinementOptions(), 1., np.ones(1), [])
    hkl = np.array([[1, 0, 0], [0, 3, 0], [0, 0, 5], [2, 0, 0], [2, 3, 1]])
    actual = DiffractionPredictor(parameters, hkl, backend="cpp").evaluate(np.asarray(parameters.start))[1]
    independent = independent_intensities(model, hkl)
    return {"arithmetic_stress": {"decimal_precision": 80, "amplitudes": amplitudes,
             "expected_intensity": expected, "cpp_intensity": float(power[0]),
             "cpp_shared_occupancy_derivative": float(jacobian[0, 0]),
             "naive_numpy_sum": float(np.sum(np.asarray(amplitudes))),
             "scope": "Artificial cancellation stress at zero phase, not a model of experimental scattering powers"},
            "physical_absences": {"spacegroup": model.spacegroup, "hkl": hkl.tolist(),
             "cpp_intensities": actual.tolist(), "gemmi_intensities": independent.tolist(),
             "maximum_forbidden_intensity": float(np.max(actual[:3]))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-worker", choices=["python", "cpp"])
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=HERE/"v08_kernel_measurements.json")
    args = parser.parse_args()
    if not kernel_status()["compiled_available"]:
        parser.error("Build the optional extension first: python tools/build_optional_kernel.py --inplace")
    with threadpool_limits(limits=1):
        if args.memory_worker:
            print(json.dumps(memory_worker(args.memory_worker, args.block_size), allow_nan=False))
            return
        result = {"schema": "ariadne_xtal.v08_kernel_validation/1",
                  "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "environment": {"python": platform.python_version(), "numpy": np.__version__,
                      "scipy": scipy.__version__, "gemmi": gemmi.__version__, "platform": platform.platform(),
                      "machine": platform.machine(), "blas_threads": 1, "kernel": kernel_status("cpp")},
                  "profile": profile_reference(),
                  "predictor": [predictor_benchmark(name) for name in
                                 ("three_atoms_three_domains", "thirty_two_atoms")],
                  "refinement": refinement_benchmark(), "cancellation": cancellation_check()}
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        result["memory"] = [json.loads(subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--memory-worker", backend,
             "--block-size", str(block_size)], env=env, text=True))
            for block_size in (1024, 8192) for backend in ("python", "cpp")]
    result["memory_interpretation"] = (
        "Separate Linux processes, one predictor evaluation after geometry/form-factor cache construction. "
        "Peak RSS includes interpreter, libraries, inputs, global real N x P Jacobian and temporary arrays. "
        "The baseline-peak difference is an observed RSS increment, not allocator-level scratch accounting. "
        "C++ removes the complex B x P derivative and complex contraction temporary; real block buffers and dense J remain."
    )
    files = ["src/ariadne_xtal/derivatives.py", "src/ariadne_xtal/refinement.py",
             "src/ariadne_xtal/kernels.py", "src/ariadne_xtal/cpp/intensity_kernel.cpp",
             "setup.py", "validation/run_v08_kernels.py"]
    result["source_sha256"] = {name: hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest() for name in files}
    result["interpretation"] = (
        "Local CPU measurements on two synthetic predictor shapes and one known-start local-refinement problem. "
        "All use the same analytic physical model; no universal speedup or replacement of BLAS/FFT is claimed."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output)
    print(json.dumps({"predictor_speedups": {row["case"]: row["median_wall_speedup"] for row in result["predictor"]},
                      "refine_speedup": result["refinement"]["median_wall_speedup"],
                      "memory": result["memory"]}, indent=2))


if __name__ == "__main__":
    main()
