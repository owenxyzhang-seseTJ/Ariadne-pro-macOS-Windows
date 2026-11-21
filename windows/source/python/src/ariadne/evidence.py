"""Append-only local hypothesis/event records; no hashes or automatic conclusions."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import uuid

from .agent.util import exclusive_lock


def append_event(job, kind, payload):
    record = {'schema': 'ariadne-event/1', 'id': uuid.uuid4().hex,
              'created': datetime.now(timezone.utc).isoformat(), 'kind': kind,
              'host': socket.gethostname(), 'software': 'Ariadne 0.8.0-native-20261108 / target DIALS 3.30.1', 'payload': payload}
    path = Path(job) / 'events.jsonl'
    with exclusive_lock(path.with_suffix('.lock'), blocking=True):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    return record['id']


def save_hypothesis(job, record):
    root = Path(job) / 'hypotheses'
    root.mkdir(exist_ok=True)
    identifier = uuid.uuid4().hex
    for parent in record.get('parents', []):
        if len(parent) != 32 or any(c not in '0123456789abcdef' for c in parent):
            raise ValueError('Invalid parent hypothesis identifier')
        if not (root / (parent + '.json')).is_file():
            raise ValueError('Missing parent hypothesis')
    value = {'schema': 'ariadne-hypothesis/1', 'id': identifier,
             'created': datetime.now(timezone.utc).isoformat(), 'state': 'proposed', **record}
    with (root / (identifier + '.json')).open('x') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    append_event(job, 'hypothesis_proposed', {'id': identifier, 'parents': record.get('parents', [])})
    return value
