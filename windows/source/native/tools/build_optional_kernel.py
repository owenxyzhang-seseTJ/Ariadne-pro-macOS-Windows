"""Build the optional local C++ kernel with the current Python interpreter.

No package installation or network access is performed. A C++17 compiler and
the development headers of this interpreter must already be available.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inplace", action="store_true", help="Build into src/ariadne_xtal for this checkout")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "setup.py", "build_ext"]
    if args.inplace:
        command.append("--inplace")
    subprocess.run(command, cwd=root, env={**os.environ, "ARIADNE_BUILD_CPP": "1"}, check=True)
    print(json.dumps({"status": "built", "python": sys.version.split()[0],
                      "inplace": args.inplace, "runtime_selector": "ARIADNE_NUMERIC_BACKEND=auto|python|cpp"}))


if __name__ == "__main__":
    main()
