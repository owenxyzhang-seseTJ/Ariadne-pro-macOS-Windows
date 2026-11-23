#!/usr/bin/env python3
"""Validate the expected contents of a crystallographic delivery package."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REQUIRED_DIRS = ("cif", "hkl", "res", "ins", "raw", "checkcif")
REQUIRED_SUFFIXES = {
    "cif": (".cif",),
    "hkl": (".hkl",),
    "res": (".res",),
    "ins": (".ins",),
    "checkcif": (".chk",),
}


def first_text(files: list[Path], suffix: str) -> str:
    for path in files:
        if path.suffix.lower() == suffix:
            return path.read_text(errors="replace")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    package = args.package.expanduser().resolve()

    if not package.is_dir():
        print("error: package must be a directory", file=sys.stderr)
        return 2

    missing_dirs = [name for name in REQUIRED_DIRS if not (package / name).is_dir()]
    missing_types: list[str] = []
    files_by_dir: dict[str, list[Path]] = {}
    for name in REQUIRED_DIRS:
        folder = package / name
        files = [item for item in folder.rglob("*") if item.is_file()] if folder.is_dir() else []
        files_by_dir[name] = files
        for suffix in REQUIRED_SUFFIXES.get(name, ()):
            if not any(item.suffix.lower() == suffix for item in files):
                missing_types.append(f"{name}/*{suffix}")

    ins_text = first_text(files_by_dir.get("ins", []), ".ins")
    res_text = first_text(files_by_dir.get("res", []), ".res")
    chk_text = first_text(files_by_dir.get("checkcif", []), ".chk")
    alerts_a = re.findall(r"^\d{3}_ALERT_\d_A\b", chk_text, flags=re.MULTILINE)
    alerts_b = re.findall(r"^\d{3}_ALERT_\d_B\b", chk_text, flags=re.MULTILINE)

    report = {
        "package": str(package),
        "missing_directories": missing_dirs,
        "missing_required_files": missing_types,
        "raw_file_count": len(files_by_dir.get("raw", [])),
        "alert_A_count": len(alerts_a),
        "alert_B_count": len(alerts_b),
        "alert_explanation_present": (
            package / "checkcif" / "alert_explanation.md"
        ).is_file(),
        "afix66_count": len(re.findall(r"^\s*AFIX\s+66\b", ins_text + res_text, re.I | re.M)),
        "omit_count": len(re.findall(r"^\s*OMIT\b", ins_text + res_text, re.I | re.M)),
        "abin_present": bool(re.search(r"^\s*ABIN\b", ins_text + res_text, re.I | re.M)),
        "fab_present": any(
            item.suffix.lower() == ".fab" for item in files_by_dir.get("hkl", [])
        ),
    }
    report["valid_minimum_layout"] = not missing_dirs and not missing_types

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0 if report["valid_minimum_layout"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
