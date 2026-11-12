#!/usr/bin/env python3
"""Create the complete source bundle with per-file SHA-256 verification.

Builds in ``dist/`` may include a platform-specific optional C++ wheel. Loose
compiler outputs and local run directories are excluded; C++ sources, tests,
report sources, open licensed fonts and scientific validation data are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

EXCLUDED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv', 'latex-build',
                 'jobs', 'raw_data', 'sessions', 'runs', 'build'}
COMPILER_SUFFIXES = {'.so', '.pyd', '.o', '.a', '.obj', '.lib', '.dll'}


def source_files(root: Path):
    for p in sorted(root.rglob('*')):
        relative = p.relative_to(root)
        if any(x in EXCLUDED_DIRS or x.endswith('.egg-info') for x in relative.parts):
            continue
        if p.is_symlink():
            raise ValueError(f'symlink in release: {relative}')
        if not p.is_file() or p.name == 'RELEASE_MANIFEST.json' or p.suffix in COMPILER_SUFFIXES:
            continue
        if p.name == '.env' or p.name.startswith('.env.') or p.name in {'runtime.json', 'core'} or p.name.startswith('core.'):
            continue
        if p.suffix in {'.sqlite', '.sqlite-wal', '.sqlite-shm', '.pem', '.key'}:
            continue
        if p.suffix in {'.pyc', '.pyo', '.aux', '.toc', '.out', '.fls', '.fdb_latexmk', '.synctex'} or p.name == '.DS_Store':
            continue
        if relative.parts[:2] == ('docs', 'v08') and p.suffix == '.log' and 'results' not in relative.parts:
            continue
        yield p, relative.as_posix()


def make(root: Path, destination: Path):
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError('ZIP output must be outside the source tree')
    files = list(source_files(root))
    records = []
    for path, name in files:
        data = path.read_bytes()
        records.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    version = tomllib.loads((root/'python/pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    native_version = tomllib.loads((root/'native/pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if native_version != version:
        raise ValueError('core and native package versions must agree')
    manifest = {'schema': 'ariadne-release-manifest/1', 'build': version+'-native-20261108',
                'version': version,
                'generated_utc': datetime.now(UTC).isoformat(), 'files': records}
    manifest_path = root/'RELEASE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path, name in files+[(manifest_path, 'RELEASE_MANIFEST.json')]:
            entry = zipfile.ZipInfo('Ariadne/'+name, (2026, 11, 8, 0, 0, 0))
            entry.external_attr = (0o100755 if path.suffix in {'.sh'} else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, path.read_bytes())
    with zipfile.ZipFile(destination) as z:
        if z.testzip():
            raise RuntimeError('ZIP CRC check failed')
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n')
    result = {'archive': str(destination), 'bytes': destination.stat().st_size,
              'files': len(files)+1, 'sha256': digest}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    make(a.root, a.destination)
