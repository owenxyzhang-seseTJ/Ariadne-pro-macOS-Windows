"""Resource-admitted local DAG execution with isolated, cancellable actions.

CPU and memory are explicit reservation tokens. Memory is not inferred RSS or
an OS memory limit. Thread environment settings constrain cooperating native
libraries; they are not a cgroup CPU quota. One parent owns graph mutations.
"""
from __future__ import annotations

from contextlib import ExitStack
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .actions import collect_artifacts
from .host_budget import HostRequest
from .planner import choose as default_choose
from .process import spawn_owned, stop_process_group, thread_environment
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
    guardian: int | None
    stdout: object
    stderr: object
    host_scope: ExitStack | None = None


@dataclass
class QueuedAction:
    node: dict
    arguments: dict
    resources: dict
    ticket: HostRequest
    scope: ExitStack
    next_poll: float = 0.


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
    def __init__(self, choose, config, observation, capabilities, experts, wake):
        self.started = time.monotonic()
        self.mailbox = queue.Queue(maxsize=1)
        def work():
            try:
                self.mailbox.put((True, choose(config, observation, capabilities, experts)))
            except BaseException as error:  # noqa: BLE001 - A dead planner thread must not strand the runner.
                self.mailbox.put((False, error))
            finally:
                wake.set()
        # Network timeouts remain bounded by planner config. A cancelled run
        # can discard its reply without letting a late decision launch work.
        self.inline = choose is default_choose and config.get('kind', 'deterministic') == 'deterministic'
        if self.inline:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            return self.mailbox.get_nowait()
        except queue.Empty:
            return None


def _launch(store, node, arguments, resources, remaining, *, queued=None, worker_module):
    work = store.begin(node['id'])
    started = time.monotonic()
    deadline = started + min(store.get('request')['budget']['action_seconds'], remaining)
    reader = writer = process = stdout = stderr = None
    try:
        atomic_json(work/'worker.input.json', {'root': str(store.root), 'node': node['id'],
                    'tool': node['tool'], 'directory': str(work), 'args': arguments,
                    'resources': resources, 'deadline': deadline,
                    'host_reservation': queued.ticket.worker_payload() if queued else None})
        environment = thread_environment(resources['cpu_threads'])
        environment['ARIADNE_MANAGED_ACTION'] = '1'
        source = str(Path(__file__).resolve().parents[2])
        python_path = [source] + [str(Path(p).resolve()) for p in environment.get('PYTHONPATH', '').split(os.pathsep) if p]
        environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(python_path))
        stdout = (work/'worker.stdout.log').open('wb')
        stderr = (work/'worker.stderr.log').open('wb')
        argv = [sys.executable, '-m', worker_module, str(work/'worker.input.json')]
        launch_options = {}
        if os.name == 'posix':
            reader, writer = os.pipe()
            argv.append(str(reader))
            launch_options['pass_fds'] = (reader,)
        process = spawn_owned(argv, cwd=work, stdout=stdout, stderr=stderr, env=environment, **launch_options)
        store.db.execute('UPDATE attempts SET worker_pid=? WHERE node=? AND number=?',
                         (process.pid, node['id'], store.node(node['id'])['attempts']))
        store.event('resources_admitted', {'node': node['id'], 'worker_pid': process.pid,
                                          **resources, 'memory_semantics': 'declared reservation, not RSS enforcement',
                                          'host_wait_seconds': started - queued.ticket.queued_at if queued else 0.})
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
    return ActiveAction(node['id'], process, work, resources, started, deadline, writer, stdout, stderr,
                        queued.scope if queued else None)


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
        if active.guardian is not None:
            os.close(active.guardian)
        active.stdout.close()
        active.stderr.close()
        if active.host_scope is not None:
            active.host_scope.close()


def _wake_on_exit(action, wake):
    def wait():
        try:
            action.process.wait()
        finally:
            wake.set()
    threading.Thread(target=wait, daemon=True).start()


def run_isolated(store, configuration, *, max_steps, choose, resolve, expand,
                 capabilities, experts, validate_inputs, host=None,
                 worker_module='ariadne.agent.action_worker'):
    request = store.get('request')
    budget = request['budget']
    pool = ResourcePool(configuration)
    active, planning, launched = {}, None, 0
    stopping = False
    frontier_changed = True
    ready_nodes = {}
    waiting = None
    wake = threading.Event()
    try:
        while True:
            wake.clear()
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
                    frontier_changed = True
            if waiting is not None:
                if stopping:
                    waiting.scope.close()
                    store.event('host_queue_left', {'node': waiting.node['id'], 'reason': 'run stopped'})
                    waiting = None
                elif now >= waiting.next_poll:
                    waiting.next_poll = now + .05
                    if waiting.ticket.try_admit():
                        key = waiting.node['id']
                        action = _launch(store, waiting.node, waiting.arguments, waiting.resources,
                                         budget['wall_seconds']-store.wall_elapsed(), queued=waiting,
                                         worker_module=worker_module)
                        pool.acquire(key, waiting.resources)
                        active[key] = action
                        _wake_on_exit(action, wake)
                        waiting = None
                        launched += 1
                        ready_nodes.pop(key, None)
            if frontier_changed:
                expand(store)
                store.propagate_failures()
                ready_nodes = {key: task_resources(store.node(key), configuration) for key in store.ready()}
                frontier_changed = False
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
                            frontier_changed = True
                        else:
                            key = decision.get('node')
                            if key not in ready_nodes:
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
                                launched += 1
                                frontier_changed = True
                            else:
                                if host is None:
                                    action = _launch(store, node, args, resources,
                                                     budget['wall_seconds']-store.wall_elapsed(),
                                                     worker_module=worker_module)
                                    pool.acquire(key, resources)
                                    active[key] = action
                                    _wake_on_exit(action, wake)
                                    launched += 1
                                    ready_nodes.pop(key)
                                else:
                                    with ExitStack() as scope:
                                        ticket = scope.enter_context(host.request(resources))
                                        waiting = QueuedAction(node, args, resources, ticket, scope.pop_all())
                                    store.event('host_queued', {'node': key, **resources,
                                                                'budget': str(host.configuration)})
            count_limit = ((max_steps is not None and launched >= max_steps)
                           or store.get('actions_used') >= budget['max_actions']
                           or store.get('decisions_used') >= budget['max_decisions'])
            if count_limit and planning is None and waiting is None:
                stopping = True
            if not stopping and planning is None and waiting is None and not frontier_changed:
                ready = [key for key, resources in ready_nodes.items() if pool.fits(resources)]
                if ready:
                    config = dict(request['planner'])
                    observation = ({'ready': ready} if choose is default_choose and config.get('kind') == 'deterministic'
                                   else {**store.snapshot(), 'ready': ready,
                                         'resource_reservations': dict(pool.allocations)})
                    config['timeout_seconds'] = min(config.get('timeout_seconds', 45),
                                                    budget['wall_seconds']-store.wall_elapsed())
                    if config['timeout_seconds'] > 0:
                        planning = PlannerCall(choose, config, observation, capabilities, experts, wake)
                        store.set('decisions_used', store.get('decisions_used')+1)
            if frontier_changed or (planning is not None and planning.inline):
                continue
            if not active and planning is None and waiting is None and (stopping or not ready_nodes):
                break
            # Worker exit and planner replies wake immediately. Other processes
            # publish cancellation and host admissions through durable storage.
            deadline = min((action.deadline for action in active.values()), default=float('inf'))
            if waiting is not None:
                deadline = min(deadline, waiting.next_poll)
            wake.wait(max(0., min(.05, deadline-time.monotonic(), budget['wall_seconds']-store.wall_elapsed())))
    finally:
        if waiting is not None:
            waiting.scope.close()
        if planning is not None:
            store.set('planning_seconds', store.get('planning_seconds', 0.) + time.monotonic()-planning.started)
            store.event('planner_reply_discarded', {'reason': 'runner exited'})
        for key, action in list(active.items()):
            _settle(store, action, {'status': 'interrupted', 'error': 'runner exited before action completion'})
            pool.release(key)
    return store.snapshot()
