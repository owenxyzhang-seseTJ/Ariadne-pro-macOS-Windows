#!/bin/sh
# 在已有的 DIALS/cctbx 环境中安装 Ariadne pro 的纯 Python 依赖并核验扩展。
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=${ARIADNE_PRO_PYTHON:-python3.12}

"$PY" - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"需要 CPython 3.12，当前为 {platform.python_version()}")
if platform.machine() != "arm64":
    raise SystemExit(f"本扩展为 arm64 构建，当前为 {platform.machine()}")
try:
    import cctbx, dials  # noqa: F401
except ImportError as error:
    raise SystemExit(
        "当前解释器缺少 DIALS/cctbx。请先准备 DIALS 3.30.1 + dxtbx 3.30.0 + cctbx 环境，"
        "再用 ARIADNE_PRO_PYTHON 指向它的 python。"
    ) from error
PY

"$PY" -m pip install --target "$ROOT/overlay" -r "$ROOT/requirements-macos-tested.txt"
PYTHONPATH="$ROOT/source/python/src:$ROOT/source/native/src:$ROOT/overlay" "$PY" - <<'PY'
from ariadne_xtal.kernels import kernel_status
status = kernel_status("cpp")
assert status["compiled_available"], status
assert status["resolved"] == "cpp", status
print("Ariadne pro macOS 已就绪：", status["build"]["compiler"], status["build"]["architecture"])
PY
