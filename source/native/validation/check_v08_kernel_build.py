"""Build and smoke-test both native distribution modes without downloading code.

Run ``python validation/check_v08_kernel_build.py`` after installing the dev
extras. Requires an existing C++17 compiler and this Python's development
headers. It creates pure-Python and platform C++ wheels plus a source archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


SMOKE = r'''
import json, os
from pathlib import Path
import numpy as np
import ariadne_xtal
from ariadne_xtal.kernels import kernel_status, resolve_backend
from ariadne_xtal.derivatives import DiffractionPredictor
from ariadne_xtal.model import Atom, Cell, Structure
from ariadne_xtal.refinement import RefinementOptions, _Parameterization
model = Structure(Cell(8.,9.,10.), [Atom("C1","C",[.12,.23,.34],u_iso=.025)])
parameters = _Parameterization(model,RefinementOptions(),1.,np.ones(1),[])
predictor = DiffractionPredictor(parameters, np.array([[1,0,0],[2,1,3]]))
_, intensity, jacobian = predictor.evaluate(np.asarray(parameters.start))
status = kernel_status()
expected = os.environ["ARIADNE_BUILD_TEST_EXPECTED"]
assert predictor.numeric_backend == expected
assert status["compiled_available"] == (expected == "cpp")
assert np.isfinite(intensity).all() and np.isfinite(jacobian).all()
assert Path(ariadne_xtal.__file__).resolve().is_relative_to(Path(os.environ["ARIADNE_BUILD_TEST_ROOT"]))
explicit_failure = None
if expected == "python":
    try:
        resolve_backend("cpp")
    except RuntimeError as error:
        explicit_failure = str(error)
    else:
        raise AssertionError("explicit cpp must fail in a pure-Python installation")
print(json.dumps({"status":status,"intensity":intensity.tolist(),"jacobian_shape":list(jacobian.shape),
                  "module_from_extracted_wheel":True,"explicit_cpp_failure":explicit_failure},allow_nan=False))
'''


def run(command, env, log_path, expected_success=True):
    completed = subprocess.run(command, cwd=PROJECT, env=env, capture_output=True, text=True)
    log_path.write_text(completed.stdout+completed.stderr)
    if (completed.returncode == 0) != expected_success:
        raise RuntimeError(f"Unexpected build result ({completed.returncode}); see {log_path}")
    return {"command": command, "returncode": completed.returncode,
            "log": str(log_path.relative_to(PROJECT))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=PROJECT.parent/"dist/v0.8")
    parser.add_argument("--output", type=Path, default=HERE/"v08_kernel_packaging.json")
    args = parser.parse_args()
    args.dist.mkdir(parents=True, exist_ok=True)
    logs = HERE/"kernel_build_logs"
    logs.mkdir(exist_ok=True)
    result = {"schema": "ariadne_xtal.v08_kernel_packaging/1", "builds": [], "wheels": []}
    base_environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                            ARIADNE_NUMERIC_BACKEND="auto")
    with tempfile.TemporaryDirectory(prefix="ariadne-kernel-build-") as temporary:
        temporary = Path(temporary)
        for choice, backend in (("0", "python"), ("1", "cpp")):
            output = temporary/backend
            output.mkdir()
            env = dict(base_environment, ARIADNE_BUILD_CPP=choice)
            command = [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(output)]
            result["builds"].append(run(command, env, logs/f"wheel_{backend}.log"))
            wheel, = output.glob("*.whl")
            unpacked = temporary/(backend+"_unpacked")
            with zipfile.ZipFile(wheel) as archive:
                members = archive.namelist()
                compiled_members = [name for name in members if name.endswith((".so", ".pyd"))]
                assert bool(compiled_members) == (backend == "cpp"), members
                archive.extractall(unpacked)
            smoke_env = dict(base_environment, PYTHONPATH=str(unpacked),
                             ARIADNE_BUILD_TEST_EXPECTED=backend, ARIADNE_BUILD_TEST_ROOT=str(unpacked))
            smoke = subprocess.run([sys.executable, "-c", SMOKE], cwd=temporary, env=smoke_env,
                                   check=True, capture_output=True, text=True)
            target = args.dist/wheel.name
            target.write_bytes(wheel.read_bytes())
            result["wheels"].append({"filename": target.name, "bytes": target.stat().st_size,
                                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                     "compiled_members": compiled_members,
                                     "smoke": json.loads(smoke.stdout)})
        # An explicit compiler choice is respected. In auto mode its failure
        # is optional; build=1 must surface the same failure to the caller.
        missing = str(temporary/"deliberately_missing_cxx")
        for choice in ("auto", "1"):
            env = dict(base_environment, ARIADNE_BUILD_CPP=choice, CXX=missing)
            build_lib = temporary/("missing_"+choice)
            command = [sys.executable, "setup.py", "build_ext", "--force", "--build-lib", str(build_lib),
                       "--build-temp", str(temporary/("objects_"+choice))]
            result["builds"].append(run(command, env, logs/f"missing_compiler_{choice}.log",
                                         expected_success=choice == "auto"))
            assert not list(build_lib.rglob("*.so")) and not list(build_lib.rglob("*.pyd"))
        env = dict(base_environment, ARIADNE_BUILD_CPP="0")
        command = [sys.executable, "-m", "build", "--no-isolation", "--sdist", "--outdir", str(args.dist)]
        result["builds"].append(run(command, env, logs/"source_archive.log"))
        source = args.dist/"ariadne_xtal-0.8.0.tar.gz"
        with tarfile.open(source) as archive:
            members = archive.getnames()
        required = ["src/ariadne_xtal/cpp/intensity_kernel.cpp", "src/ariadne_xtal/kernels.py",
                    "tools/build_optional_kernel.py", "setup.py", "tests/test_kernels_v08.py"]
        assert all(any(name.endswith("/"+relative) for name in members) for relative in required)
        result["sdist"] = {"filename": source.name, "bytes": source.stat().st_size,
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "required_source_members_verified": required}
    sources = ["setup.py", "pyproject.toml", "MANIFEST.in", "tools/build_optional_kernel.py",
               "src/ariadne_xtal/cpp/intensity_kernel.cpp", "validation/check_v08_kernel_build.py"]
    result["source_sha256"] = {name: hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in sources}
    result["interpretation"] = (
        "Both wheels were imported from separate extracted directories, excluding the editable native source path. "
        "The compiled wheel is specific to this Python ABI/platform; the pure wheel is portable Python. "
        "No dependency installation or compiler download occurred. Source archive contains buildable C++ source."
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output)
    print(json.dumps({"wheels": [row["filename"] for row in result["wheels"]],
                      "sdist": result["sdist"]["filename"],
                      "build_exit_codes": [row["returncode"] for row in result["builds"]]}, indent=2))


if __name__ == "__main__":
    main()
