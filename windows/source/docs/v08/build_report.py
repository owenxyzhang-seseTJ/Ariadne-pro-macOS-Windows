"""Compile the two illustrated LaTeX reports with bundled open font subsets.

Usage: python docs/v08/build_report.py
Requires XeLaTeX on PATH. Figures are distributed as PDFs; regenerating them
is optional and uses tools/make_figures.py in the installed Ariadne environment.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    executable = shutil.which("xelatex")
    if not executable:
        raise SystemExit("Install XeLaTeX (TeX Live or MiKTeX) before building the reports.")
    for name in ("Ariadne_0_8_0_Textbook", "Ariadne_0_8_0_Loop"):
        # A third pass resolves a table of contents whose length changed.
        for iteration in range(3):
            result = subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                check=False,
            )
            (root / (name + ".build.log")).write_text(result.stdout, encoding="utf-8")
            if result.returncode:
                raise SystemExit(f"LaTeX failed: see {name}.build.log")
        log = (root / (name + ".log")).read_text(encoding="utf-8", errors="replace")
        failures = ("Missing character:", "There were undefined references", "Overfull \\hbox")
        if any(marker in log for marker in failures):
            raise SystemExit(f"Typesetting requires review: see {name}.log")
        print(root / (name + ".pdf"))


if __name__ == "__main__":
    main()
