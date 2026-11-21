"""Run the numerical and runtime regression suite used by the platform matrix."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NUMERICS = [
    "native/tests/test_core.py", "native/tests/test_kernels_v08.py", "native/tests/test_joint_v08.py",
    "native/tests/test_refinement.py", "native/tests/test_refinement_v08.py",
    "native/tests/test_phasing_v08.py", "tests/test_calibrated_pixel_inputs.py",
    "native/tests/test_count_refinement.py", "tests_rebuild/test_reduction_v08.py",
    "tests_rebuild/test_lattice_selection.py", "tests_rebuild/test_feedback.py",
]
RUNTIME = ["tests_rebuild/test_runtime_upgrade.py", "tests_rebuild/test_agent.py", "tests_rebuild/test_host_budget.py",
           "tests_rebuild/test_scheduler_v08.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-native", action="store_true")
    parser.add_argument("--dependency-path", type=Path, action="append", default=[])
    parser.add_argument("--backend", choices=["cpp", "python"], default="cpp")
    parser.add_argument("--junitxml", type=Path)
    parser.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    source = ROOT / ("windows/source" if os.name == "nt" else "macos/source")
    paths = [source / name for name in ("python/src", "skills/ariadne-reduction/scripts", "tests")]
    if not args.installed_native:
        paths.insert(0, source / "native/src")
    paths.extend(path.resolve() for path in args.dependency_path)
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, paths)),
                       ARIADNE_PLUGIN_ROOT=str(source), ARIADNE_NUMERIC_BACKEND=args.backend,
                       PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                       MKL_NUM_THREADS="1", ARIADNE_CPU_THREADS="1")
    tests = list(NUMERICS)
    if not args.numerics_only:
        tests.extend(RUNTIME)
        if os.name != "nt":
            tests.append("tests/test_tool_lifecycle.py")
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
               "-c", str(source / "python/pyproject.toml"), *tests]
    if args.junitxml:
        command += ["--junitxml", str(args.junitxml.resolve())]
    return subprocess.call(command, cwd=source, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
