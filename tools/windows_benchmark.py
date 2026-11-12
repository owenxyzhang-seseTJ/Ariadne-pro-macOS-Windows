"""原生 Windows 能力与性能实测；复用数值基准函数，不调用发布验证入口。"""
import argparse
import ast
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


def peak_rss_bytes():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def benchmark_functions(source):
    # 仅加载数值基准定义及其依赖导入；排除 Unix resource 和发布记录代码。
    path = source / "native/validation/run_v08_kernels.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {"problem", "sequence", "predictor_benchmark", "independent_intensities", "refinement_benchmark"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases = [a for a in node.names if a.name not in {"resource", "hashlib"}]
            if aliases:
                selected.append(ast.Import(names=aliases))
        elif isinstance(node, ast.ImportFrom):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            selected.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"__name__": "windows_numeric_benchmark", "__file__": str(path)}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-backend", choices=("python", "cpp"))
    args = parser.parse_args()
    if sys.platform != "win32":
        raise RuntimeError("此驱动必须从原生 Windows Python 运行")
    import numpy as np
    from threadpoolctl import threadpool_limits, threadpool_info
    from ariadne_xtal.derivatives import DiffractionPredictor
    from ariadne_xtal.kernels import kernel_status
    bench = benchmark_functions(args.source)
    if args.memory_backend:
        baseline = peak_rss_bytes()
        parameters, values, hkl, laws = bench["problem"]("eighty_atoms_memory")
        prediction = DiffractionPredictor(parameters, hkl, laws, backend=args.memory_backend).evaluate(values)
        print(json.dumps({"backend": args.memory_backend, "peak_rss_bytes": peak_rss_bytes(),
            "baseline_peak_rss_bytes": baseline, "returned_jacobian_bytes": prediction[2].nbytes,
            "intensity_sum": float(np.sum(prediction[1]))}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"product": "Ariadne pro Windows", "version": "0.8.0", "variant": "win-x86_64-localopt1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
                        "packages": {}, "kernel": kernel_status()},
        "predictor": [], "memory": [], "boundaries": {}, "acceptance": {}}
    for package in ("numpy", "scipy", "gemmi", "threadpoolctl", "pytest", "mcp", "dials", "cctbx-base"):
        try:
            report["environment"]["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["environment"]["packages"][package] = None
    target = args.output / "measurements.json"
    def save():
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with threadpool_limits(limits=1):
        report["environment"]["threadpools"] = threadpool_info()
        for name in ("three_atoms_three_domains", "thirty_two_atoms", "eighty_atoms_memory"):
            print("数值基准："+name, flush=True)
            report["predictor"].append(bench["predictor_benchmark"](name, repeats=5))
            save()
        print("局部精修性能", flush=True)
        report["refinement"] = bench["refinement_benchmark"]()
        save()
        for backend in ("python", "cpp"):
            completed = subprocess.run([sys.executable, str(Path(__file__)), "--source", str(args.source),
                "--output", str(args.output), "--memory-backend", backend], capture_output=True,
                text=True, encoding="utf-8", check=True, timeout=180)
            report["memory"].append(json.loads(completed.stdout))
        print("公开实测 FeN4 重精修", flush=True)
        real = load_module("fen4_fixture", args.source / "native/validation/run_real_fen4.py")
        from ariadne_xtal.refinement import refine, RefinementOptions
        from ariadne_xtal.io import read_structure, write_structure, read_reflections, write_reflections
        model, observations = real.load_case()
        start = time.perf_counter()
        fitted = refine(model, observations, RefinementOptions(weight_a=.0282, weight_b=.3122,
                        weight_cycles=20, covariance_scale="reduced_chi2"))
        report["measured_fen4"] = {"seconds": time.perf_counter()-start, "n_reflections": len(observations),
            "success": bool(fitted.success), "metrics": fitted.to_dict()["metrics"], "doi": "10.1107/S2056989018012161",
            "scope": "公开合并实测强度上的已知模型重精修，不是从头解结构或原始图像还原"}
        write_structure(fitted.structure, args.output / "fen4-refined.cif")
        assert fitted.success and fitted.metrics["jacobian_rank"] == 10 and len(observations) == 71
        for metric, expected, tolerance in (("r1_gt", .0403, 5e-5), ("r1_all", .0413, 5e-5),
                ("wr2", .0815, 5e-5), ("gof", 1.183, 5e-4)):
            assert abs(fitted.metrics[metric]-expected) <= tolerance
        save()
        print("中文及空格路径往返", flush=True)
        directory = args.output / "中文 空格路径"
        directory.mkdir()
        write_structure(fitted.structure, directory / "模型.cif")
        write_structure(fitted.structure, directory / "模型.json")
        write_reflections(observations, directory / "反射.json")
        cif = directory / "模型.cif"
        cif.write_bytes(cif.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        recovered = read_structure(cif)
        assert len(recovered.atoms) == len(fitted.structure.atoms)
        assert np.allclose([a.frac for a in recovered.atoms], [a.frac for a in fitted.structure.atoms], atol=1e-6)
        assert len(read_structure(directory / "模型.json").atoms) == len(fitted.structure.atoms)
        assert np.array_equal(read_reflections(directory / "反射.json").hkl, observations.hkl)
        report["unicode_paths"] = {"success": True, "directory": str(directory), "cif_crlf": True}
        print("P1 合成解相位与精修", flush=True)
        phase = load_module("phasing_fixture", args.source / "native/validation/run_phasing_recovery.py")
        start = time.perf_counter()
        phase_result = phase.run_case(args.output / "phasing")
        report["phasing"] = {"seconds": time.perf_counter()-start, "n_reflections": phase_result["n_reflections"],
            "observations_unchanged": phase_result["observations_unchanged"],
            "candidates": [{"rank": c["rank"], "success": c["refinement"]["success"],
                "rms_cartesian_angstrom": c["after_refinement"]["rms_cartesian_angstrom"],
                "wr2": c["refinement"]["metrics"]["wr2"]} for c in phase_result["results"]]}
        assert phase_result["observations_unchanged"] and len(phase_result["results"]) == 4
        assert all(c["success"] and c["rms_cartesian_angstrom"] < 1e-5 and c["wr2"] < 1e-6
                   for c in report["phasing"]["candidates"])
        save()
    for name, code in {
            "dials_runtime": "import dials, dxtbx, cctbx; print(dials.__file__); print(dxtbx.__file__)",
            "dials_spotfinding": "from dials.algorithms.image.threshold import DispersionThreshold; print(DispersionThreshold)",
            "parent_mcp_import": "import ariadne.server",
            "persistent_agent_execution_lock": "from pathlib import Path; from ariadne.agent.util import exclusive_lock; exec(\"with exclusive_lock(Path('windows-compatibility.lock')):\\n pass\")",
            "persistent_agent_help": "from ariadne.agent.cli import main; main(['--help'])",
            "native_capabilities": "from ariadne_xtal.cli import main; raise SystemExit(main(['capabilities']))"}.items():
        outcome = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60)
        report["boundaries"][name] = {"returncode": outcome.returncode, "stdout": outcome.stdout[-16000:],
                                      "stderr": outcome.stderr[-8000:]}
    report["acceptance"] = {
        "native_windows": sys.platform == "win32",
        "compiled_cpp_available": report["environment"]["kernel"]["compiled_available"],
        "predictor_scaled_jacobian_below_1e_10": all(r["maximum_scaled_jacobian_difference"] < 1e-10 for r in report["predictor"]),
        "refinement_converged": all(r["success"] for r in report["refinement"]["runs"]),
        "refinement_prediction_sigma_rms_below_1e_6": report["refinement"]["prediction_difference_sigma_rms"] < 1e-6,
        "measured_fen4_regression": True, "synthetic_phasing_recovery": True, "unicode_path_roundtrip": True}
    save()
    print(json.dumps({"output": str(target), "acceptance": report["acceptance"]}, ensure_ascii=False, indent=2), flush=True)
    if not all(report["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
