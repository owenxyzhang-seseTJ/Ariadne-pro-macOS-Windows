"""Update the Windows source mirror from the maintained source tree."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("native/src", "native/tests", "python/src", "skills/ariadne-reduction/scripts",
               "tests", "tests_rebuild", "benchmarks")
SOURCE_SUFFIXES = {".py", ".cpp", ".h", ".hpp"}


def sync(check=False):
    source = ROOT / "macos/source"
    mirror = ROOT / "windows/source"
    changed = []
    for directory in SOURCE_DIRS:
        for path in sorted((source / directory).rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            relative = path.relative_to(source)
            target = mirror / relative
            if target.is_file() and target.read_bytes() == path.read_bytes():
                continue
            changed.append(relative.as_posix())
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    settings = parser.parse_args()
    changes = sync(settings.check)
    if changes:
        print("\n".join(changes))
    print(f"{len(changes)} {'differing' if settings.check else 'updated'} source files")
    raise SystemExit(bool(changes) if settings.check else 0)
