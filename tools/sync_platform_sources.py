"""Update the Windows source mirror from the maintained source tree."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("native/src", "native/tests", "python/src", "skills/ariadne-reduction/scripts",
               "tests", "tests_rebuild", "benchmarks")
SOURCE_SUFFIXES = {".py", ".cpp", ".h", ".hpp"}
SKILL_SUFFIXES = {".md", ".py", ".sh", ".yaml"}
PLUGIN_FILES = (".codex-plugin/plugin.json", ".agents/plugins/marketplace.json",
                ".mcp.json", "skills/ariadne-reduction/SKILL.md")


def sync(check=False):
    source = ROOT / "macos/source"
    mirror = ROOT / "windows/source"
    changed = []
    paths = [path for directory in SOURCE_DIRS for path in (source / directory).rglob("*")
             if path.suffix in SOURCE_SUFFIXES and path.is_file()]
    paths.extend(path for path in (source / "skills/solve-single-crystal").rglob("*")
                 if path.suffix in SKILL_SUFFIXES and path.is_file())
    paths.extend(source / name for name in PLUGIN_FILES)
    for path in sorted(paths):
        relative = path.relative_to(source)
        target = mirror / relative
        if target.is_file() and target.read_bytes() == path.read_bytes():
            continue
        changed.append(relative.as_posix())
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
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
