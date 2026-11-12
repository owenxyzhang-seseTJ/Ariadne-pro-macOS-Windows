"""Actual isolated workers, native subprocesses and durable DAG scheduling."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ariadne.agent import engine
from ariadne.agent.spec import validate_request
from ariadne.agent.store import Store

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='Local process-group supervisor is POSIX')


def _session(tmp_path, configuration=None, nodes=None, budget=None, executables=None):
    data = tmp_path/'input.json'
    data.write_text('{"value": 7}')
    request = {'mode': 'custom', 'inputs': {'data': str(data)},
               'scheduler': {'max_workers': 2, 'cpu_threads': 2, 'memory_mb': 512,
                             **(configuration or {})},
               'nodes': nodes or [engine.node(key, 'native.run', {'request': {'action': 'capabilities'}})
                                 for key in ('a', 'b')]}
    if budget:
        request['budget'] = budget
    if executables:
        request['executables'] = executables
    root = tmp_path/'session'
    engine.initialize(request, root)
    return root


def _state(root):
    store = Store(root)
    try:
        return ({n['id']: store.node(n['id']) for n in store.summaries()},
                [dict(row) for row in store.db.execute('SELECT * FROM attempts ORDER BY started')])
    finally:
        store.close()


def _overlap(attempts):
    a, b = attempts[:2]
    return max(a['started'], b['started']) < min(a['finished'], b['finished'])


def test_real_native_workers_overlap_respect_dependencies_and_resume(tmp_path):
    nodes = [engine.node(key, 'native.run', {'request': {'action': 'capabilities'}}) for key in ('a', 'b')]
    nodes += engine.completion_nodes(['a', 'b'])
    root = _session(tmp_path, nodes=nodes)
    first = engine.run(root, max_steps=2)
    states, attempts = _state(root)
    assert states['a']['status'] == states['b']['status'] == 'succeeded'
    assert states['review']['status'] == 'pending'
    assert _overlap(attempts)
    assert first['cumulative_action_seconds'] > first['wall_elapsed_seconds']
    finished = engine.run(root)
    assert finished['actions_used'] == 4
    assert all(n['status'] == 'succeeded' for n in finished['nodes'])
    assert engine.run(root)['actions_used'] == 4
    states, attempts = _state(root)
    assert states['a']['attempts'] == states['b']['attempts'] == 1
    review = next(a for a in attempts if a['node'] == 'review')
    assert review['started'] >= max(a['finished'] for a in attempts if a['node'] in {'a', 'b'})
    command = json.loads((root/'attempts/a/001/native.command.json').read_text())
    assert set(command['thread_environment'].values()) == {'1'}
    store = Store(root)
    assert store.verify_events()
    store.close()


@pytest.mark.parametrize('configuration', [{'cpu_threads': 1}, {'memory_mb': 256}])
def test_cpu_or_declared_memory_tokens_serialize_ready_jobs(tmp_path, configuration):
    root = _session(tmp_path, configuration)
    result = engine.run(root)
    _, attempts = _state(root)
    assert all(n['status'] == 'succeeded' for n in result['nodes'])
    assert not _overlap(attempts)


def test_failed_dependencies_are_explicitly_skipped_and_can_be_reviewed(tmp_path):
    nodes = [engine.node('bad', 'artifact.inspect', {'artifact': engine.input_ref('data'), 'json_pointer': '/missing'}),
             engine.node('dependent', 'artifact.inspect', {'artifact': engine.input_ref('data')}, ['bad']),
             *engine.completion_nodes(['dependent'])]
    root = _session(tmp_path, nodes=nodes)
    result = engine.run(root)
    states, attempts = _state(root)
    assert states['bad']['status'] == 'failed'
    assert states['dependent']['status'] == 'skipped' and states['dependent']['attempts'] == 0
    assert states['review']['status'] == states['package']['status'] == 'succeeded'
    assert result['actions_used'] == 3
    assert not any(a['node'] == 'dependent' for a in attempts)
    with pytest.raises(ValueError, match='consumed'):
        engine.retry(root, 'bad')


def test_last_decision_budget_is_executed_and_action_budget_survives_resume(tmp_path):
    root = _session(tmp_path, budget={'max_actions': 1, 'max_decisions': 1})
    result = engine.run(root)
    assert result['actions_used'] == result['decisions_used'] == 1
    assert sum(n['status'] == 'succeeded' for n in result['nodes']) == 1
    assert engine.run(root)['actions_used'] == 1


@pytest.mark.parametrize('configuration', [{'max_workers': 0}, {'cpu_threads': True},
                                           {'memory_mb': -1}, {'isolate_actions': False}])
def test_invalid_resource_budgets_fail_before_execution(configuration):
    with pytest.raises(ValueError):
        validate_request({'mode': 'custom', 'scheduler': {'max_workers': 2, **configuration},
                          'nodes': [engine.node('a', 'native.run', {'request': {'action': 'capabilities'}})]})


@pytest.mark.parametrize('resources', [{'cpu_threads': 3}, {'memory_mb': 513}])
def test_task_larger_than_total_budget_is_rejected_without_scheduling(tmp_path, resources):
    with pytest.raises(ValueError, match='exceeds'):
        _session(tmp_path, nodes=[engine.node('a', 'native.run', {'request': {'action': 'capabilities'}},
                                               resources=resources)])


def test_failed_worker_launch_is_persisted_as_failed_not_unsettled(tmp_path, monkeypatch):
    from ariadne.agent import scheduler
    root = _session(tmp_path)
    def cannot_launch(*args, **kwargs):
        raise OSError('injected launch failure')
    monkeypatch.setattr(scheduler.subprocess, 'Popen', cannot_launch)
    with pytest.raises(OSError, match='injected launch failure'):
        engine.run(root)
    states, attempts = _state(root)
    assert states['a']['status'] == attempts[0]['status'] == 'failed'
    assert states['b']['status'] == 'pending'
    assert 'injected launch failure' in states['a']['error']


def test_dynamic_candidate_expansion_is_committed_once_by_parent(tmp_path):
    # These are executable protocol fixtures, not installed SHELXT/SHELXL.
    from create_demo import write_model
    from test_native_protocol import INS, executable_fixture
    cif = tmp_path/'model.cif'
    write_model(cif)
    solve = executable_fixture(tmp_path/'solve_fixture',
        'from pathlib import Path\n'
        + f"Path('job_a.res').write_text({INS!r})\nPath('job_b.res').write_text({INS!r})\n"
        + "Path('job_a.hkl').write_text('1 0 0 111 3\\n')\nPath('job_b.hkl').write_text('1 0 0 222 4\\n')\n")
    refine = executable_fixture(tmp_path/'refine_fixture',
        'from pathlib import Path\n'
        + "Path('job.res').write_text(Path('job.ins').read_text())\n"
        + f"Path('job.cif').write_text({cif.read_text()!r})\n")
    ins, hkl = tmp_path/'input.ins', tmp_path/'input.hkl'
    ins.write_text(INS)
    hkl.write_text('1 0 0 999 9\n')
    root = tmp_path/'session'
    engine.initialize({'mode': 'solve', 'inputs': {'ins': str(ins), 'hkl': str(hkl)},
                       'executables': {'shelxt': solve, 'shelxl': refine},
                       'scheduler': {'max_workers': 2, 'cpu_threads': 2}}, root)
    result = engine.run(root)
    assert all(n['status'] == 'succeeded' for n in result['nodes']), result
    states, attempts = _state(root)
    assert states['solve']['expanded'] == 1
    assert _overlap([a for a in attempts if a['node'].startswith('refine_')])
    assert (root/'attempts/refine_00/001/job.hkl').read_text().split()[3] == '111'
    assert (root/'attempts/refine_01/001/job.hkl').read_text().split()[3] == '222'
    assert len({n['id'] for n in result['nodes']}) == len(result['nodes'])
    assert engine.run(root)['actions_used'] == result['actions_used']


def _nested_python(tmp_path):
    """Protocol fixture: real native CLI plus a long-lived sibling grandchild."""
    marker = tmp_path/'descendants.json'
    wrapper = tmp_path/'scientific_python'
    wrapper.write_text('#!'+sys.executable+'\n'
                       'import subprocess,sys,time,os,json\nfrom pathlib import Path\n'
                       "native=subprocess.Popen([sys.executable,*sys.argv[1:]])\n"
                       "sleeper=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                       + f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),native.pid,sleeper.pid]))\n"
                       + 'native.wait()\ntime.sleep(60)\n')
    wrapper.chmod(0o755)
    return wrapper, marker


def _alive(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        # A zombie no longer computes or holds ordinary process resources;
        # orphan reaping is owned by the platform's init process.
        if text.rsplit(')', 1)[1].strip().split()[0] == 'Z':
            return False
    except FileNotFoundError:
        if sys.platform.startswith('linux'):
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _await(predicate, seconds=12):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.025)
    raise AssertionError('subprocess condition did not complete within the test budget')


def test_isolated_deadline_stops_native_cli_and_nested_descendants(tmp_path):
    wrapper, marker = _nested_python(tmp_path)
    root = _session(tmp_path, {'isolate_actions': True, 'max_workers': 1, 'cpu_threads': 1},
                    nodes=[engine.node('slow', 'native.run', {'request': {'action': 'capabilities'}})],
                    budget={'action_seconds': 1.5, 'wall_seconds': 15},
                    executables={'scientific_python': str(wrapper)})
    result = engine.run(root)
    assert result['nodes'][0]['status'] == 'failed'
    assert 'time budget' in result['nodes'][0]['error']
    pids = json.loads(marker.read_text())
    _await(lambda: not any(_alive(pid) for pid in pids))


@pytest.mark.parametrize('crash', [False, True])
def test_cancel_or_parent_crash_reclaims_native_tree_and_lock(tmp_path, crash):
    wrapper, marker = _nested_python(tmp_path)
    root = _session(tmp_path, nodes=[engine.node('slow', 'native.run', {'request': {'action': 'capabilities'}})],
                    budget={'action_seconds': 30, 'wall_seconds': 60},
                    executables={'scientific_python': str(wrapper)})
    with (tmp_path/'runner.stdout').open('wb') as stdout, (tmp_path/'runner.stderr').open('wb') as stderr:
        runner = subprocess.Popen([sys.executable, '-m', 'ariadne.agent.cli', 'run', str(root)],
                                   stdout=stdout, stderr=stderr)
        try:
            _await(marker.is_file)
            pids = json.loads(marker.read_text())
            with pytest.raises(RuntimeError, match='another runner'):
                engine.run(root)
            if crash:
                runner.kill()
            else:
                engine.cancel(root)
            runner.wait(timeout=5)
            _await(lambda: not any(_alive(pid) for pid in pids))
            if crash:
                with pytest.raises(RuntimeError, match='unsettled'):
                    engine.run(root)
                engine.recover(root, 'slow', confirmed_process_stopped=True)
                assert _state(root)[0]['slow']['status'] == 'interrupted'
            else:
                assert _state(root)[0]['slow']['status'] == 'cancelled'
                engine.resume(root)
            assert _state(root)[0]['slow']['attempts'] == 1
        finally:
            if runner.poll() is None:
                runner.kill()
                runner.wait()


def test_cancel_discards_late_planner_decision_without_launch(tmp_path, monkeypatch):
    root = _session(tmp_path)
    def late(*args):
        time.sleep(.2)
        return {'action': 'run', 'node': 'a'}
    monkeypatch.setattr(engine, 'choose', late)
    import threading
    timer = threading.Timer(.05, lambda: engine.cancel(root))
    timer.start()
    try:
        result = engine.run(root)
        time.sleep(.25)
        assert result['actions_used'] == engine.status(root)['actions_used'] == 0
    finally:
        timer.join()
