"""交错比较保存的原版和本机优化版，并直接比较预测输出。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from threadpoolctl import threadpool_limits


def worker(source, output, probe):
    spec = importlib.util.spec_from_file_location("bench", source / "native/validation/run_v08_kernels.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    import ariadne_xtal
    from ariadne_xtal.derivatives import DiffractionPredictor
    with threadpool_limits(limits=1):
        result = {"module_path": ariadne_xtal.__file__, "predictor": [], "refinement": None}
        arrays = {}
        for case in ("three_atoms_three_domains", "thirty_two_atoms", "eighty_atoms_memory"):
            result["predictor"].append(bench.predictor_benchmark(case, repeats=3))
            if probe:
                parameters, values, hkl, laws = bench.problem(case)
                _, intensity, jacobian = DiffractionPredictor(parameters, hkl, laws, backend="cpp").evaluate(values)
                arrays[case + "_intensity"] = intensity
                arrays[case + "_jacobian"] = jacobian
        result["refinement"] = bench.refinement_benchmark()
        output.write_text(json.dumps(result, indent=2) + "\n")
        if probe:
            np.savez(output.with_suffix(".npz"), **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.worker_output:
        worker(args.source, args.worker_output, args.probe)
        return
    runs = []
    for index, mode in enumerate(("baseline", "optimized", "optimized", "baseline")):
        print(f"交错对照 {index + 1}/4：{mode}", flush=True)
        env = dict(os.environ)
        if mode == "baseline":
            env["PYTHONPATH"] = str(args.directory / "baseline-source") + os.pathsep + env.get("PYTHONPATH", "")
        output = args.directory / f"paired-{index + 1}-{mode}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source),
                   "--directory", str(args.directory), "--worker-output", str(output)]
        if index < 2:
            command.append("--probe")
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(completed.stderr + completed.stdout)
        result = json.loads(output.read_text())
        expected_root = args.directory / "baseline-source" if mode == "baseline" else args.source / "native/src"
        assert Path(result["module_path"]).is_relative_to(expected_root)
        runs.append({"mode": mode, "result": result})
    comparison = {"order": [run["mode"] for run in runs], "predictor": [], "refinement": {}}
    for case in runs[0]["result"]["predictor"]:
        row = {"case": case["case"], "n_atoms": case["n_atoms"], "n_reflections": case["n_reflections"],
               "n_domains": case["n_domains"], "backends": {}}
        for backend in ("python", "cpp"):
            timings = {}
            for mode in ("baseline", "optimized"):
                values = [sample["wall_seconds"] / item["evaluations_per_run"]
                    for run in runs if run["mode"] == mode
                    for item in run["result"]["predictor"] if item["case"] == case["case"]
                    for sample in item["runs"] if sample["backend"] == backend]
                timings[mode] = {"round_mean_seconds": values, "median_seconds": float(np.median(values))}
            before, after = [timings[mode]["median_seconds"] for mode in ("baseline", "optimized")]
            row["backends"][backend] = {"timings": timings, "speedup": before/after,
                                        "time_reduction_percent": (1-after/before)*100}
        comparison["predictor"].append(row)
    for backend in ("python", "cpp"):
        timings = {}
        for mode in ("baseline", "optimized"):
            rows = [sample for run in runs if run["mode"] == mode
                    for sample in run["result"]["refinement"]["runs"] if sample["backend"] == backend]
            assert all(row["success"] and row["actual_backend"] == backend for row in rows)
            timings[mode] = {"wall_seconds": [row["wall_seconds"] for row in rows],
                             "median_seconds": float(np.median([row["wall_seconds"] for row in rows])),
                             "chi_squared": [row["chi_squared"] for row in rows],
                             "jacobian_ranks": [row["jacobian_rank"] for row in rows]}
        before, after = [timings[mode]["median_seconds"] for mode in ("baseline", "optimized")]
        comparison["refinement"][backend] = {"timings": timings, "speedup": before/after,
                                             "time_reduction_percent": (1-after/before)*100}
    comparison["direct_output_comparison"] = []
    left = args.directory / "paired-1-baseline.npz"
    right = args.directory / "paired-2-optimized.npz"
    with np.load(left) as original, np.load(right) as optimized:
        for name in original.files:
            reference, actual = original[name], optimized[name]
            scale = np.maximum(np.max(np.abs(reference), axis=0), 1.)
            scaled = float(np.max(np.abs(reference-actual)/scale))
            assert np.allclose(reference, actual, rtol=1e-11, atol=2e-9)
            assert scaled < 1e-11
            comparison["direct_output_comparison"].append({"array": name, "shape": list(reference.shape),
                "maximum_absolute_difference": float(np.max(np.abs(reference-actual))),
                "maximum_scaled_difference": scaled})
    # 临时大数组已完成逐元素比较，保留统计值和可复现驱动。
    left.unlink()
    right.unlink()
    target = args.directory / "paired-comparison.json"
    target.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
