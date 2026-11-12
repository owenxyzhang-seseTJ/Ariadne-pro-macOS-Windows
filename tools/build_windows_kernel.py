"""用隔离的 LLVM-MinGW 工具链为当前 Windows CPython 构建原生扩展。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--compiler", type=Path, required=True)
args = parser.parse_args()
if sys.platform != "win32" or sys.maxsize <= 2**32:
    raise RuntimeError("必须从原生 Windows x64 Python 运行本工具")
include = Path(sys.base_prefix) / "include"
if not (include / "Python.h").is_file():
    include = Path(sysconfig.get_path("include"))
library = Path(sys.base_prefix) / "libs" / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
if not library.is_file():
    raise FileNotFoundError(f"当前解释器缺少开发导入库：{library}")
output = args.source / "native/src/ariadne_xtal" / ("_intensity_cpp" + sysconfig.get_config_var("EXT_SUFFIX"))
if output.exists():
    raise FileExistsError(f"目标已存在，停止以保护已有二进制：{output}")
compiler_command = [str(args.compiler.resolve())]
if args.compiler.name.lower() == "zig.exe":
    compiler_command += ["c++", "-target", "x86_64-windows-gnu"]
command = compiler_command + ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off",
           "-shared", "-static-libstdc++", "-I" + str(include),
           str(args.source / "native/src/ariadne_xtal/cpp/intensity_kernel.cpp"), str(library),
           "-o", str(output)]
print(json.dumps({"compiler": str(args.compiler), "python": sys.executable, "command": command}, indent=2), flush=True)
subprocess.run(command, check=True)
print(json.dumps({"built_extension": str(output), "bytes": output.stat().st_size}), flush=True)
