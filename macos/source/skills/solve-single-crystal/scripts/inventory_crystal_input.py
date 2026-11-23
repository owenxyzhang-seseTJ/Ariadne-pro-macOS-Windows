#!/usr/bin/env python3
"""Read-only inventory and SFRM classification for crystallographic inputs."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import zipfile
from pathlib import Path


HEADER_BYTES = 15 * 512
CALIBRATION_WORDS = ("ACTIVE MASK", "INITIAL BACKGROUND", "BACKGROUND")


def parse_sfrm_type(data: bytes) -> str:
    text = data[:HEADER_BYTES].decode("latin-1", errors="ignore")
    for offset in range(0, len(text), 80):
        record = text[offset : offset + 80]
        if record.startswith("TYPE") and ":" in record:
            return record.split(":", 1)[1].strip()
    return "UNKNOWN"


def collect(path: Path) -> tuple[list[str], dict[str, str]]:
    names: list[str] = []
    frame_types: dict[str, str] = {}
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            for name in names:
                if name.lower().endswith(".sfrm"):
                    with archive.open(name) as handle:
                        frame_types[name] = parse_sfrm_type(handle.read(HEADER_BYTES))
        return names, frame_types
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        names = [str(item.relative_to(path)) for item in files]
        for item, name in zip(files, names):
            if item.suffix.lower() == ".sfrm":
                with item.open("rb") as handle:
                    frame_types[name] = parse_sfrm_type(handle.read(HEADER_BYTES))
        return names, frame_types
    raise ValueError("input must be a directory or ZIP archive")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        names, frame_types = collect(args.input.expanduser().resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    extensions = collections.Counter(
        Path(name).suffix.lower() or "[no extension]" for name in names
    )
    calibration = {
        name: kind
        for name, kind in frame_types.items()
        if any(word in kind.upper() for word in CALIBRATION_WORDS)
    }
    diffraction = {
        name: kind for name, kind in frame_types.items() if name not in calibration
    }
    programs = {
        name: shutil.which(name)
        for name in ("shelxt", "shelxl", "platon", "saint", "sadabs", "xprep")
    }
    report = {
        "input": str(args.input.expanduser().resolve()),
        "file_count": len(names),
        "extensions": dict(sorted(extensions.items())),
        "sfrm_count": len(frame_types),
        "sfrm_calibration_count": len(calibration),
        "sfrm_diffraction_candidate_count": len(diffraction),
        "sfrm_types": frame_types,
        "core_files": [
            name
            for name in names
            if Path(name).suffix.lower()
            in {".raw", ".p4p", ".hkl", ".abs", ".prp", ".ins", ".res", ".cif", ".mol"}
        ],
        "programs": programs,
        "reintegration_possible_from_inventory": bool(diffraction),
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"Input: {report['input']}")
    print(f"Files: {report['file_count']}")
    print("Extensions:")
    for suffix, count in report["extensions"].items():
        print(f"  {suffix}: {count}")
    print(
        "SFRM: "
        f"{len(frame_types)} total; {len(calibration)} calibration/background; "
        f"{len(diffraction)} diffraction candidates"
    )
    for name, kind in frame_types.items():
        print(f"  {name}: {kind}")
    if frame_types and not diffraction:
        print("WARNING: SFRM files are calibration/background only; no diffraction "
              "sequence was identified.")
    print("Programs:")
    for name, location in programs.items():
        print(f"  {name}: {location or 'not found'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
