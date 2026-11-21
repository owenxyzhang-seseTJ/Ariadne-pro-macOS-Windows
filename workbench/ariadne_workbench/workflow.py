"""Persist the workbench plan in the same DAG and run store used by MCP."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import os
import sys
import time
import uuid

from ariadne.agent import async_runs, engine
from ariadne.agent.backends import DIALS_PROCESS_OPTIONS
from ariadne.agent.host_budget import HostBudget, validate_resources
from ariadne.agent.spec import ToolSpec
from ariadne.agent.store import Store
from ariadne.agent.util import artifact, artifact_path, atomic_json, exclusive_lock, read_json


SCHEMA = 'ariadne-workbench-dag/1'
ACTION = 'workbench.step'
WORKER = 'ariadne_workbench.workflow'
TOOL_SPEC = {ACTION: ToolSpec(frozenset({'action', 'project', 'settings', 'follow_refine'}),
                             frozenset({'previous'}), 'Execute one registered desktop calculation')}
STATES = {'pending': 'queued', 'succeeded': 'completed', 'blocked': 'failed'}
CONTROL_FILES = {'worker.input.json', 'worker.output.json', 'worker.stdout.log', 'worker.stderr.log',
                 'workbench-result.json'}


def project_lock(folder):
    return exclusive_lock(folder/'.workbench.lock', blocking=True)


def _resources(action, settings, host):
    max_cpu = host.capacity()['cpu_threads'] if host is not None else os.cpu_count() or 1
    stage = action.split(':', 1)[1] if action.startswith('dials:') else None
    cpu = min(4, max_cpu) if stage == 'find_spots' else 1
    option = DIALS_PROCESS_OPTIONS.get(stage)
    if option:
        value = settings.get('dials_options', {}).get(option)
        if value is not None:
            if type(value) is not int or value < 1:
                raise ValueError(f'{option} 必须是正整数。')
            cpu = value
    memory = 3072 if stage == 'integrate' else 512 + 384*cpu if stage == 'find_spots' else 512
    resources = validate_resources(settings.get('resources', {'cpu_threads': cpu, 'memory_mb': memory}))
    if host is not None:
        host._check_fits(resources, host.capacity())
    return resources


def submit(jobfolder):
    return async_runs.submit(jobfolder/'session', uuid.uuid4().hex,
                             runner_module=WORKER, worker_module=WORKER,
                             import_root=Path(__file__).resolve().parents[1])


def initialize(jobfolder, project, job):
    host = HostBudget.configured()
    actions = [step['action'] for step in job['steps']]
    nodes = []
    for index, action in enumerate(actions):
        key = f'step-{index:02d}'
        args = dict(action=action, project=engine.input_ref('project'), settings=job['settings'],
                    follow_refine=(job['action'] == 'automatic' and index > 0 and action == 'legacy:shelxl-maps'))
        dependencies = []
        if index:
            previous = nodes[-1]['id']
            args['previous'] = engine.ref(previous, '/changes')
            dependencies.append(previous)
        nodes.append(engine.node(key, ACTION, args, dependencies, resources=_resources(action, job['settings'], host)))
    source = jobfolder/'project-input.json'
    atomic_json(source, project)
    try:
        engine.initialize(dict(mode='custom', experts=[], inputs={'project': str(source)}, nodes=nodes,
                               scheduler={'cpu_threads': max(node['resources']['cpu_threads'] for node in nodes),
                                          'isolate_actions': True},
                               budget={'wall_seconds': 864000., 'action_seconds': 86400., 'max_attempts': 20,
                                       'max_actions': 20*len(nodes), 'max_decisions': 40*len(nodes)}),
                          jobfolder/'session', tools=TOOL_SPEC, worker_module=WORKER)
    finally:
        source.unlink(missing_ok=True)
    job['workflow'] = SCHEMA
    atomic_json(jobfolder/'job.json', job)


def advance_project(folder, project):
    job_id = project.get('lastJob')
    if not job_id:
        return project
    jobfolder = folder/'jobs'/job_id
    session = jobfolder/'session'
    if not (session/'agent.sqlite').is_file():
        return project
    cursor = project.get('workflowCursor', {})
    applied = cursor.get('completed', 0) if cursor.get('job') == job_id else 0
    with closing(Store(session)) as store:
        for position,node in enumerate(store.summaries()):
            if position < applied:
                continue
            if node['status'] != 'succeeded':
                break
            result = store.node(node['id'])['result']
            project.update(result['updates'])
            actions = [*project.get('completedActions', []), result['action']]
            project['completedActions'] = list(dict.fromkeys(actions))
            project['workflowCursor'] = dict(job=job_id, completed=position+1)
    if project.get('workflowCursor', {}) != cursor:
        project['updated'] = time.time()
        atomic_json(folder/'project.json', project)
    return project


def load_project(folder):
    return advance_project(folder, read_json(folder/'project.json'))


def _tail(path, limit):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size-limit))
        return stream.read(limit).decode('utf-8', errors='replace')


def status(jobfolder, job):
    session = jobfolder/'session'
    run = async_runs.latest(session)
    steps = []
    log = ''
    errors = []
    warnings = []
    with closing(Store(session)) as store:
        cancel_requested = store.get('cancel_requested')
        resources = {}
        for event in store.db.execute(
                "SELECT payload FROM events WHERE kind IN ('host_queued','resources_admitted') ORDER BY seq"):
            item = json.loads(event['payload'])
            resources[item['node']] = item
        for previous,node in zip(job['steps'], store.summaries()):
            attempt = store.db.execute('SELECT * FROM attempts WHERE node=? AND number=?',
                                       (node['id'],node['attempts'])).fetchone()
            step = dict(action=previous['action'], status=STATES.get(node['status'], node['status']),
                        attempt=node['attempts'], nodeId=node['id'])
            if node['id'] in resources:
                entry = resources[node['id']]
                step['resources'] = {name: entry[name] for name in ('cpu_threads', 'memory_mb')}
                if 'host_wait_seconds' in entry:
                    step['hostWaitSeconds'] = entry['host_wait_seconds']
            if attempt is not None:
                out = session/attempt['directory']
                step['elapsedSeconds'] = node['elapsed']
                if node['status'] == 'succeeded':
                    result = store.node(node['id'])['result']
                    warnings.extend(result['updates'].get('refinementWarnings', []))
                    step['artifacts'] = [(session/item['path']).relative_to(jobfolder.parents[1]).as_posix()
                                         for item in result['artifacts']]
                for name in ('run.log', 'worker.stderr.log', 'worker.stdout.log'):
                    log = (log + _tail(out/name, 12000))[-8000:]
            if node['error'] and node['status'] in {'failed', 'blocked', 'interrupted'}:
                step['error'] = node['error']
                errors.append(node['error'])
            steps.append(step)
    current = next((i for i,step in enumerate(steps) if step['status'] != 'completed'), len(steps)-1)
    if run is not None and run['status'] in async_runs.ACTIVE:
        state = 'running' if any(s['status'] == 'running' for s in steps) else 'queued'
        message = '正在回收任务进程。' if cancel_requested else None
    elif run is None or run['status'] == 'interrupted':
        state, message = 'interrupted', '运行已中断。可从未完成的步骤继续。'
    elif run['status'] == 'cancelled':
        state, message = 'cancelled', '任务已取消，已完成步骤和记录已保留。'
    elif all(s['status'] == 'completed' for s in steps):
        state, message = 'completed', '计算已完成。可查看计算结果和科学诊断。'
        if warnings:
            message += ' '.join(warnings)
    else:
        state = 'failed'
        message = errors[0] if errors else run.get('error') or '任务未完成。请检查运行预算和日志。'
    if run is not None:
        log = (log + _tail(session/f"run-{run['id']}.log", 8000))[-8000:]
    return {**job, 'steps': steps, 'status': state, 'current': current, 'message': message,
            'logTail': log, 'runId': run['id'] if run is not None else None,
            'pid': run['pid'] if run is not None else None,
            'cancelPending': bool(cancel_requested and state in {'queued', 'running'})}


def resume(jobfolder):
    root = jobfolder/'session'
    with closing(Store(root)) as store:
        nodes = store.summaries()
        running = {row['node']: row['worker_pid'] for row in store.db.execute(
            "SELECT node,worker_pid FROM attempts WHERE status='running'")}
    if running:
        from ariadne.server import pid_alive

        if any(pid and pid_alive(pid) for pid in running.values()):
            raise ValueError('前一轮工作进程仍在退出。')
        for key in running:
            engine.recover(root, key, confirmed_process_stopped=True)
    for node in nodes:
        if node['status'] in {'running', 'failed', 'cancelled', 'interrupted', 'blocked'}:
            engine.retry(root, node['id'])
    engine.resume(root)
    return submit(jobfolder)


def execute_step(ctx, tool, arguments):
    """Application-owned actions run inside the scheduler's isolated process."""
    if tool != ACTION:
        raise ValueError('Unregistered workbench action')
    from .runner import execute_step as calculate

    folder = ctx.root.parents[2]
    project = read_json(artifact_path(ctx.root, arguments['project']))
    upstream = arguments.get('previous', {})
    project.update(upstream)
    settings = {**arguments['settings'], '_admitted_cpu_threads': ctx.resources['cpu_threads']}
    action = arguments['action']
    with (ctx.directory/'run.log').open('w', encoding='utf-8', buffering=1) as log:
        updates = calculate(action, project, folder, ctx.directory, settings, log,
                            follow_refine=arguments['follow_refine'])
    if updates.get('structure') and not action.startswith(('dials:', 'analysis:', 'legacy:')):
        from .science import read_structure

        updates['cell'] = read_structure(folder/updates['structure']).cell.to_dict()
    files = [artifact(ctx.root, path, 'workbench_output')
             for path in sorted(ctx.directory.rglob('*'))
             if path.is_file() and path.name not in CONTROL_FILES]
    result = dict(action=action, updates=updates, changes={**upstream, **updates},
                  artifacts=files, scientific_acceptance='not_established')
    path = ctx.directory/'workbench-result.json'
    atomic_json(path, result)
    return {**result, 'result_artifact': artifact(ctx.root, path, 'scientific_result')}


if __name__ == '__main__':
    # Only the local application supplies these module paths to the shared runtime.
    if len(sys.argv) == 4 and sys.argv[1] == '--run':
        async_runs._worker(Path(sys.argv[2]), sys.argv[3], worker_module=WORKER)
    else:
        from ariadne.agent.action_worker import main

        raise SystemExit(main(execute_action=execute_step))
