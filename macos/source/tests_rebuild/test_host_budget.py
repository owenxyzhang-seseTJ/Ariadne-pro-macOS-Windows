"""Cross-process admission, cancellation, and recovery at the shared host boundary."""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ariadne.agent import engine
from ariadne.agent.host_budget import HostBudget
from ariadne.agent.store import Store


def _wait(predicate, seconds=25):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.03)
    raise AssertionError('the expected cross-process state was not reached')


def _budget(tmp_path, monkeypatch, *, cpu=2, memory=256):
    config = tmp_path / 'host resources.json'
    config.write_text(json.dumps({'cpu_threads': cpu, 'memory_mb': memory}), encoding='utf-8')
    monkeypatch.setenv('ARIADNE_HOST_BUDGET', str(config))
    return HostBudget(config)


def test_fifo_checks_both_cpu_and_memory_and_removes_cancelled_waiter(tmp_path, monkeypatch):
    host = _budget(tmp_path, monkeypatch)
    with host.request({'cpu_threads': 1, 'memory_mb': 192}) as first:
        assert first.try_admit()
        with host.request({'cpu_threads': 1, 'memory_mb': 128}) as queued:
            assert not queued.try_admit()
            with host.request({'cpu_threads': 1, 'memory_mb': 32}) as later:
                assert not later.try_admit()
        with host.request({'cpu_threads': 1, 'memory_mb': 32}) as next_request:
            assert next_request.try_admit()
            with pytest.raises(ValueError, match='cpu_threads exceeds'):
                with host.request({'cpu_threads': 3, 'memory_mb': 32}):
                    pytest.fail('oversized request entered the queue')
    assert host._load() == []
    assert sorted(p.name for p in host.directory.iterdir()) == ['queue.json', 'queue.lock']


def test_live_worker_retains_resources_after_owner_is_killed(tmp_path, monkeypatch):
    host = _budget(tmp_path, monkeypatch, cpu=1, memory=128)
    marker = tmp_path / 'token'
    owner_code = '''
import sys
import time
from pathlib import Path
from ariadne.agent.host_budget import HostBudget
with HostBudget(sys.argv[1]).request({"cpu_threads": 1, "memory_mb": 128}) as ticket:
    assert ticket.try_admit()
    Path(sys.argv[2]).write_text(ticket.token)
    time.sleep(60)
'''
    owner = subprocess.Popen([sys.executable, '-c', owner_code, str(host.configuration), str(marker)])
    worker = None
    started = tmp_path / 'worker-ready'
    try:
        token = _wait(lambda: marker.read_text() if marker.exists() and marker.stat().st_size == 32 else None)
        worker_code = '''
import sys
import time
from pathlib import Path
from ariadne.agent.host_budget import HostBudget
with HostBudget(sys.argv[1]).worker(sys.argv[2]):
    Path(sys.argv[3]).touch()
    time.sleep(60)
'''
        worker = subprocess.Popen([sys.executable, '-c', worker_code, str(host.configuration),
                                   token, str(started)])
        _wait(started.exists)
        owner.kill()
        owner.wait(timeout=10)
        with host.request({'cpu_threads': 1, 'memory_mb': 128}) as ticket:
            assert not ticket.try_admit()
            worker.kill()
            worker.wait(timeout=10)
            assert ticket.try_admit()
    finally:
        for process in (owner, worker):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert host._load() == []


def test_two_scientific_sessions_wait_for_one_host_slot(tmp_path, monkeypatch):
    host = _budget(tmp_path, monkeypatch, cpu=1, memory=128)
    fixture = Path(__file__).resolve().parents[1] / 'native/validation/public_data/2242624.cif'
    jobs = [tmp_path / 'first project', tmp_path / 'second project']
    for job in jobs:
        engine.initialize({'mode': 'audit', 'experts': [], 'inputs': {'cif': str(fixture)},
                           'scheduler': {'default_task_memory_mb': 128},
                           'budget': {'max_decisions': 1, 'action_seconds': 20, 'wall_seconds': 45}}, job)
    processes = []
    try:
        with host.request({'cpu_threads': 1, 'memory_mb': 128}) as barrier:
            assert barrier.try_admit()
            for job in jobs:
                code = 'from ariadne.agent.engine import run; import sys; run(sys.argv[1], max_steps=1)'
                processes.append(subprocess.Popen([sys.executable, '-c', code, str(job)]))
            _wait(lambda: len(host._load()) == 3)
            assert [engine.status(job)['actions_used'] for job in jobs] == [0, 0]
        for process in processes:
            assert process.wait(timeout=25) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    for job in jobs:
        store = Store(job)
        try:
            node = store.node('audit')
            assert node['status'] == 'succeeded'
            started = [row for row in store.db.execute("SELECT payload FROM events WHERE kind='resources_admitted'")]
            assert len(started) == 1 and json.loads(started[0]['payload'])['host_wait_seconds'] > 0
        finally:
            store.close()
    assert host._load() == []
