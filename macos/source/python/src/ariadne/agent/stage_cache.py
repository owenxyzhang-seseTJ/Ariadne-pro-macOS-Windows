"""Bounded local cache for deterministic scientific stages."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from .util import digest, dumps, exclusive_lock


class StageCache:
    def __init__(self, directory: Path, max_bytes: int):
        self.directory = directory
        self.max_bytes = max_bytes
        directory.mkdir(parents=True, exist_ok=True)

    def get(self, dependencies, compute):
        key = digest(dependencies)
        # A fixed set of lock files keeps cache metadata bounded.
        with exclusive_lock(self.directory / 'locks' / key[:2], blocking=True):
            with closing(sqlite3.connect(self.directory / 'cache.sqlite', timeout=10)) as db:
                db.execute('''CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL,
                    logical_bytes INTEGER NOT NULL, last_used INTEGER NOT NULL)''')
                row = db.execute('SELECT payload, checksum FROM entries WHERE key=?', (key,)).fetchone()
                if row is not None and digest(row[0]) == row[1]:
                    with db:
                        db.execute('UPDATE entries SET last_used=? WHERE key=?', (time.time_ns(), key))
                    return json.loads(row[0]), True

                result = compute()
                payload = dumps(result)
                size = len(payload.encode('utf-8'))
                with db:
                    db.execute('DELETE FROM entries WHERE key=?', (key,))
                    if size <= self.max_bytes:
                        db.execute('INSERT INTO entries VALUES (?, ?, ?, ?, ?)',
                                   (key, payload, digest(payload), size, time.time_ns()))
                    total = db.execute('SELECT coalesce(sum(logical_bytes), 0) FROM entries').fetchone()[0]
                    if total > self.max_bytes:
                        for other, used in db.execute('SELECT key, logical_bytes FROM entries ORDER BY last_used, key').fetchall():
                            db.execute('DELETE FROM entries WHERE key=?', (other,))
                            total -= used
                            if total <= self.max_bytes:
                                break
                return result, False
