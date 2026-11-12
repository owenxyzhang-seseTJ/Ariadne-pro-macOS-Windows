"""Resource-admitted local DAG execution with isolated, cancellable actions.

CPU and memory are explicit reservation tokens. Memory is not inferred RSS or
an OS memory limit. Thread environment settings constrain cooperating native
libraries; they are not a cgroup CPU quota. One parent owns graph mutations.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .actions import collect_artifacts
from .process import stop_process_group, thread_environment
from .spec import task_resources
from .util import artifact_path, atomic_json, read_json


@dataclass
class ActiveAction:
    key: str
    process: subprocess.Popen
    directory: Path
    resources: dict
    started: float
    deadline: float
    guardian: int
    stdout: object
    stderr: object


class ResourcePool:
    """Exact bookkeeping of requested resources; no unmeasured usage claims."""
    def __init__(self, configuration):
        self.configuration = configuration
        self.allocations = {}

    def fits(self, resources):
        config = self.configuration
        return (len(self.allocations) < config['max_workers']
                and sum(r['cpu_threads'] for r in self.allocations.values()) + resources['cpu_threads'] <= config['cpu_threads']
                and (config['memory_mb'] is None or sum(r['memory_mb'] for r in self.allocations.values())
                     + resources['memory_mb'] <= config['memory_mb']))

    def acquire(self, key, resources):
        if key in self.allocations or not self.fits(resources):
            raise RuntimeError('resource admission invariant violated')
        self.allocations[key] = dict(resources)

    def release(self, key):
        del self.allocations[key]


class PlannerCall:
    """One isolated decision over an immutable snapshot; no SQLite in thread."""
    def __init__(self, choose, config, observation, capabilities, experts):
        self.started = time.monotonic()
        self.mailbox = queue.Queue(maxsize=1)
        def work():
            try:
                self.mailbox.put((True, choose(config, observation, capabilities, experts)))
            except BaseException as error:  # noqa: BLE001 - A dead planner thread must not strand the runner.
                self.mailbox.put((False, error))
        # Network timeouts remain bounded by planner config. A cancelled run
        # can discard its reply without letting a late decision launch work.
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def poll(self):
        try:
            return self.mailbox.get_nowait()
        except queue.Empty:
            return None


def _launch(store, node, arguments, resources, remaining):
    work = store.begin(node['id'])
    started = time.monotonic()
    deadline = started + min(store.get('request')['budget']['action_seconds'], remaining)
    reader = writer = process = stdout = stderr = None
    try:
        atomic_json(work/'worker.input.json', {'root': str(store.root), 'node': node['id'],
                    'tool': node['tool'], 'directory': str(work), 'args': arguments,
                    'resources': resources, 'deadline': deadline})
        environment = thread_environment(resources['cpu_threads'])
        environment['ARIADNE_MANAGED_ACTION'] = '1'
        source = str(Path(__file__).resolve().parents[2])
        python_path = [source] + [str(Path(p).resolve()) for p in environment.get('PYTHONPATH', '').split(os.pathsep) if p]
        environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(python_path))
        reader, writer = os.pipe()
        stdout = (work/'worker.stdout.log').open('wb')
        stderr = (work/'worker.stderr.log').open('wb')
        process = subprocess.Popen([sys.executable, '-m', 'ariadne.agent.action_worker',
                                    str(work/'worker.input.json'), str(reader)], cwd=work,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   env=environment, start_new_session=True, pass_fds=(reader,))
        store.db.execute('UPDATE attempts SET worker_pid=? WHERE node=? AND number=?',
                         (process.pid, node['id'], store.node(node['id'])['attempts']))
        store.event('resources_admitted', {'node': node['id'], 'worker_pid': process.pid,
                                          **resources, 'memory_semantics': 'declared reservation, not RSS enforcement'})
    except BaseException as error:
        if process is not None:
            stop_process_group(process)
        if writer is not None:
            os.close(writer)
        for stream in (stdout, stderr):
            if stream is not None:
                stream.close()
        store.finish(node['id'], 'failed', time.monotonic()-started,
                     error=f'action worker could not be launched: {type(error).__name__}: {error}')
        raise
    finally:
        if reader is not None:
            os.close(reader)
    return ActiveAction(node['id'], process, work, resources, started, deadline, writer, stdout, stderr)


def _settle(store, active, forced=None):
    try:
        stop_process_group(active.process)
        if forced is not None:
            output = forced
        elif active.process.returncode != 0:
            output = {'status': 'failed', 'error': f'action worker exited {active.process.returncode}; inspect worker logs'}
        else:
            output = read_json(active.directory/'worker.output.json')
            if output.get('status') not in {'succeeded', 'failed', 'blocked', 'interrupted', 'cancelled'}:
                raise ValueError('invalid worker completion record')
            if output['status'] == 'succeeded':
                for record in collect_artifacts(output['result']):
                    artifact_path(store.root, record)
        store.finish(active.key, output['status'], time.monotonic()-active.started,
                     result=output.get('result'), error=output.get('error'))
    except Exception as error:  # noqa: BLE001 - Corrupt/missing completion evidence becomes a failed attempt.
        if store.node(active.key)['status'] == 'running':
            store.finish(active.key, 'failed', time.monotonic()-active.started,
                         error=f'{type(error).__name__}: {error}')
    finally:
        os.close(active.guardian)
        active.stdout.close()
        active.stderr.close()


def run_isolated(store, configuration, *, max_steps, choose, resolve, expand,
                 capabilities, experts, validate_inputs):
    if os.name != 'posix':
        raise RuntimeError('isolated local DAG scheduling requires Linux, macOS or WSL')
    request = store.get('request')
    budget = request['budget']
    pool = ResourcePool(configuration)
    active, planning, launched = {}, None, 0
    stopping = False
    try:
        while True:
            now = time.monotonic()
            cancelled = bool(store.get('cancel_requested'))
            wall_left = budget['wall_seconds']-store.wall_elapsed()
            if cancelled or wall_left <= 0:
                stopping = True
            for key, action in list(active.items()):
                forced = None
                if cancelled:
                    forced = {'status': 'cancelled', 'error': 'session cancellation requested'}
                elif wall_left <= 0 or now >= action.deadline:
                    forced = {'status': 'failed', 'error': 'TimeoutError: scientific action exceeded its time budget'}
                elif sum((action.directory/name).stat().st_size for name in ('worker.stdout.log', 'worker.stderr.log')) > 256*1024*1024:
                    forced = {'status': 'failed', 'error': 'action worker logs exceeded the storage budget'}
                if forced is not None or action.process.poll() is not None:
                    _settle(store, action, forced)
                    pool.release(key)
                    del active[key]
            expand(store)
            store.propagate_failures()
            if planning is not None:
                reply = planning.poll()
                if stopping or reply is not None:
                    store.set('planning_seconds', store.get('planning_seconds', 0.) + time.monotonic()-planning.started)
                    planning = None
                    if stopping:
                        store.event('planner_reply_discarded', {'reason': 'session stopped before decision admission'})
                    else:
                        valid, decision = reply
                        if not valid:
                            store.event('planner_failed', {'error': str(decision)})
                            raise decision
                        store.event('planner_decision', decision)
                        if decision['action'] == 'pause':
                            stopping = True
                        elif decision['action'] == 'propose':
                            proposed = decision['node']
                            validate_inputs(proposed.get('args', {}), store.get('inputs'))
                            if len(store.summaries()) >= budget['max_actions']:
                                raise RuntimeError('proposed graph exceeds action budget')
                            store.add_nodes([proposed])
                        else:
                            key = decision.get('node')
                            if key not in store.ready():
                                raise ValueError('planner selected a node that is not ready')
                            node = store.node(key)
                            resources = task_resources(node, configuration)
                            if not pool.fits(resources):
                                raise ValueError('planner selected a node outside the admitted resource frontier')
                            try:
                                args = resolve(node['args'], store, node['dependencies'])
                            except Exception as error:  # noqa: BLE001 - Resolution failures belong to this node.
                                store.begin(key)
                                store.finish(key, 'failed', 0., error=f'{type(error).__name__}: {error}')
                            else:
                                action = _launch(store, node, args, resources, budget['wall_seconds']-store.wall_elapsed())
                                pool.acquire(key, resources)
                                active[key] = action
                            launched += 1
            count_limit = ((max_steps is not None and launched >= max_steps)
                           or store.get('actions_used') >= budget['max_actions']
                           or store.get('decisions_used') >= budget['max_decisions'])
            if count_limit and planning is None:
                stopping = True
            if not stopping and planning is None:
                observation = store.snapshot()
                ready = [key for key in observation['ready'] if pool.fits(task_resources(store.node(key), configuration))]
                if ready:
                    observation['ready'] = ready
                    observation['resource_reservations'] = dict(pool.allocations)
                    config = dict(request['planner'])
                    config['timeout_seconds'] = min(config.get('timeout_seconds', 45),
                                                    budget['wall_seconds']-store.wall_elapsed())
                    if config['timeout_seconds'] > 0:
                        planning = PlannerCall(choose, config, observation, capabilities, experts)
                        store.set('decisions_used', store.get('decisions_used')+1)
            if not active and planning is None and (stopping or not store.ready()):
                break
            time.sleep(.02)
    finally:
        if planning is not None:
            store.set('planning_seconds', store.get('planning_seconds', 0.) + time.monotonic()-planning.started)
            store.event('planner_reply_discarded', {'reason': 'runner exited'})
        for key, action in list(active.items()):
            _settle(store, action, {'status': 'interrupted', 'error': 'runner exited before action completion'})
            pool.release(key)
    return store.snapshot()
