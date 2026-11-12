"""Optional C++17 extension; all Python capabilities remain installable alone.

ARIADNE_BUILD_CPP=0 disables compilation.  The default 'auto' attempts an
optional build.  ARIADNE_BUILD_CPP=1 requires a successful compiler invocation.
No NumPy headers, pybind11, downloaded code or architecture-specific flags are
needed.  Runtime selection is independent: ARIADNE_NUMERIC_BACKEND.
"""
from __future__ import annotations

import os
import shlex
import shutil
import sys
import sysconfig

from setuptools import Extension, setup


choice = os.environ.get("ARIADNE_BUILD_CPP", "auto").lower()
if choice not in {"auto", "0", "1"}:
    raise RuntimeError("ARIADNE_BUILD_CPP must be auto, 0 or 1")
extensions = []
if choice != "0":
    # Relocated Python distributions sometimes retain an unavailable build-
    # machine compiler in sysconfig. Preserve explicit user choices, otherwise
    # find an already installed compiler instead of assuming its filesystem path.
    if sys.platform != "win32":
        for variable, candidates in (("CC", ("cc", "gcc", "clang")),
                                     ("CXX", ("c++", "g++", "clang++"))):
            configured = shlex.split(sysconfig.get_config_var(variable) or "")
            if variable not in os.environ and (not configured or shutil.which(configured[0]) is None):
                available = next((shutil.which(name) for name in candidates if shutil.which(name)), None)
                if available:
                    os.environ[variable] = available
    flags = (["/O2", "/std:c++17", "/fp:precise"] if sys.platform == "win32" else
             ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"])
    extensions.append(Extension(
        "ariadne_xtal._intensity_cpp",
        sources=["src/ariadne_xtal/cpp/intensity_kernel.cpp"],
        include_dirs=[sysconfig.get_path("include")],
        language="c++", extra_compile_args=flags, optional=choice == "auto",
    ))

setup(ext_modules=extensions)
