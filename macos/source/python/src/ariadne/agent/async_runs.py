"""Detached runs for MCP clients, stored transactionally in the session database."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import uuid

from . import engine
from .store import Store
from .util import LockBusy, exclusive_lock, now


ACTIVE = ('queued', 'running')


def _schema(store):
    store.db.executescript('''
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, request_key TEXT NOT NULL UNIQUE, max_steps INTEGER,
            status TEXT NOT NULL, pid INTEGER, created TEXT NOT NULL, updated TEXT NOT NULL,
            error TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_run ON runs((1))
            WHERE status IN ('queued', 'running');
    ''')


def _get(store, run_id):
    row = store.db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
    if row is None:
        raise ValueError('unknown run id')
    return dict(row)


def _mark(store, run_id, status, error=None, pid=None):
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        store.db.execute('UPDATE runs SET status=?,updated=?,error=?,pid=coalesce(?,pid) WHERE id=?',
                         (status, now(), error, pid, run_id))
        store.event('async_run_' + status, {'run_id': run_id, 'error': error})


def _lost(row):
    pid = row['pid']
    if pid is None:
        return True
    if os.name == 'nt':
        from .windows_job import pid_alive
    else:
        from ..server import pid_alive
    return not pid_alive(pid)


def submit(session, request_key, max_steps=None, *, runner_module='ariadne.agent.async_runs',
           worker_module=engine.DEFAULT_ACTION_WORKER, import_root=None):
    if not isinstance(request_key, str) or not request_key.strip() or len(request_key) > 128:
        raise ValueError('request_key must be a nonempty string of at most 128 characters')
    if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= 10000):
        raise ValueError('max_steps must be null or an integer in 1..10000')
    root = Path(session).expanduser().resolve()
    with exclusive_lock(root / '.async.lock', blocking=True):
        store = Store(root)
        try:
            if store.get('action_worker_module', engine.DEFAULT_ACTION_WORKER) != worker_module:
                raise ValueError('This session requires its application executor')
            _schema(store)
            row = store.db.execute('SELECT * FROM runs WHERE request_key=?', (request_key,)).fetchone()
            if row is not None:
                if row['max_steps'] != max_steps:
                    raise ValueError('request_key was already used with a different max_steps')
                return _refresh(store, dict(row))
            active = store.db.execute("SELECT * FROM runs WHERE status IN ('queued','running')").fetchone()
            if active is not None and _refresh(store, dict(active))['status'] in ACTIVE:
                raise LockBusy('another asynchronous run is active for this session')
            if store.get('cancel_requested'):
                raise RuntimeError('resume the cancelled session before submitting a new run')
            with exclusive_lock(root / '.runner.lock'):
                run_id = uuid.uuid4().hex
                at = now()
                with store.db:
                    store.db.execute('BEGIN IMMEDIATE')
                    store.db.execute('INSERT INTO runs VALUES (?, ?, ?, ?, NULL, ?, ?, NULL)',
                                     (run_id, request_key, max_steps, 'queued', at, at))
                    store.event('async_run_queued', {'run_id': run_id, 'max_steps': max_steps})
            try:
                detached = ({'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
                            if os.name == 'nt' else {'start_new_session': True})
                environment = os.environ.copy()
                source = str(Path(__file__).resolve().parents[2])
                paths = [source, *(str(Path(p).resolve()) for p in environment.get('PYTHONPATH', '').split(os.pathsep) if p)]
                if import_root is not None:
                    paths.insert(0, str(Path(import_root).resolve()))
                environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
                command = [sys.executable, '-m', runner_module]
                if runner_module != 'ariadne.agent.async_runs':
                    command.append('--run')
                with (root / f'run-{run_id}.log').open('wb') as log:
                    child = subprocess.Popen(
                        [*command, str(root), run_id],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=root, env=environment, **detached)
            except OSError as error:
                _mark(store, run_id, 'failed', f'runner could not be launched: {error}')
            else:
                store.db.execute('UPDATE runs SET pid=? WHERE id=?', (child.pid, run_id))
                # Reap stdio-owned supervisors promptly on POSIX. The runner itself
                # is detached and continues if the MCP client closes.
                from threading import Thread
                Thread(target=child.wait, daemon=True).start()
            return _get(store, run_id)
        finally:
            store.close()


def _refresh(store, run):
    if run['status'] in ACTIVE and _lost(run):
        _mark(store, run['id'], 'interrupted', 'runner exited before recording a terminal status')
        return _get(store, run['id'])
    return run


def status(session, run_id):
    root = Path(session).expanduser().resolve()
    with exclusive_lock(root / '.async.lock', blocking=True):
        store = Store(root)
        try:
            _schema(store)
            return _refresh(store, _get(store, run_id))
        finally:
            store.close()


def latest(session):
    """Return the most recently submitted run, including its recovered liveness."""
    root = Path(session).expanduser().resolve()
    with exclusive_lock(root / '.async.lock', blocking=True):
        store = Store(root)
        try:
            _schema(store)
            row = store.db.execute('SELECT * FROM runs ORDER BY rowid DESC LIMIT 1').fetchone()
            return _refresh(store, dict(row)) if row is not None else None
        finally:
            store.close()


def events(session, after=0, limit=100):
    if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('after must be a nonnegative integer and limit must be in 1..200')
    from json import loads

    store = Store(Path(session).expanduser())
    try:
        rows = [dict(row) for row in store.db.execute(
            'SELECT seq,at,kind,payload FROM events WHERE seq>? ORDER BY seq LIMIT ?', (after, limit + 1))]
        for row in rows[:limit]:
            row['payload'] = loads(row['payload'])
        return {'events': rows[:limit], 'next_cursor': rows[min(len(rows), limit) - 1]['seq'] if rows else after,
                'more': len(rows) > limit}
    finally:
        store.close()


def cancel(session, run_id):
    root = Path(session).expanduser().resolve()
    with exclusive_lock(root / '.async.lock', blocking=True):
        store = Store(root)
        try:
            _schema(store)
            row = _refresh(store, _get(store, run_id))
            if row['status'] in ACTIVE:
                engine.cancel(root)
                row['cancel_requested'] = True
            return row
        finally:
            store.close()


def _worker(root, run_id, *, worker_module=engine.DEFAULT_ACTION_WORKER):
    store = Store(root)
    try:
        with exclusive_lock(root / '.async.lock', blocking=True):
            row = _get(store, run_id)
            # A supervisor may be interrupted between launch and persisting PID.
            if row['status'] != 'queued':
                return
            _mark(store, run_id, 'running', pid=os.getpid())
        status, error = 'completed', None
        try:
            engine.run(root, max_steps=row['max_steps'], force_isolated=True, worker_module=worker_module)
        except Exception as exc:
            status, error = 'failed', f'{type(exc).__name__}: {exc}'
        with exclusive_lock(root / '.async.lock', blocking=True):
            if store.get('cancel_requested'):
                status = 'cancelled'
            _mark(store, run_id, status, error)
    finally:
        store.close()


if __name__ == '__main__':
    _worker(Path(sys.argv[1]), sys.argv[2])
