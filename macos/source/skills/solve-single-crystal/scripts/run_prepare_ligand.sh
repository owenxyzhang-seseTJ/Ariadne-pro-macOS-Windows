#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

try_python() {
  local candidate="$1"
  local resolved
  shift
  [ -n "$candidate" ] || return 1
  if [[ "$candidate" == */* ]]; then
    [ -x "$candidate" ] || return 1
    resolved="$candidate"
  else
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    [ -n "$resolved" ] || return 1
  fi
  if "$resolved" -c 'import rdkit' >/dev/null 2>&1; then
    exec "$resolved" "$script_dir/prepare_ligand.py" "$@"
  fi
  return 1
}

if [ -n "${RDKIT_PYTHON:-}" ]; then
  try_python "$RDKIT_PYTHON" "$@" || true
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
  try_python "$VIRTUAL_ENV/bin/python" "$@" || true
fi

try_python python3 "$@" || true
try_python python "$@" || true
try_python "${HOME}/.venv/bin/python" "$@" || true

printf '%s\n' \
  'error: no Python interpreter with RDKit was found.' \
  'Set RDKIT_PYTHON to a suitable interpreter and rerun.' >&2
exit 3
