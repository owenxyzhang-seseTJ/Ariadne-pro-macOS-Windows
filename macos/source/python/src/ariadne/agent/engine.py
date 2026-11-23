"""Observe -> choose -> execute -> inspect -> persist -> expand the real graph."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import time
from typing import Any
import zipfile

from .actions import Context, collect_artifacts, execute
from .backends import ANALYSES, DIALS_OPTIONS, plugin_root
from .planner import choose
from .process import Cancelled, NeedsReview
from .spec import TOOLS, check_name, task_resources, validate_request, validate_scheduler
from .store import Store
from .util import artifact, artifact_path, atomic_json, content_id, dumps, exclusive_lock, pointer, read_json, snapshot_input


BUILD = '0.8.0-resolution'
DEFAULT_ACTION_WORKER = 'ariadne.agent.action_worker'


def ref(node: str, path: str = ''):
    return {'$ref': {'node': node, 'pointer': path}}


def input_ref(name: str):
    return {'$input': name}


def node(key: str, tool: str, args: dict, dependencies=(), allow_failed=False, resources=None):
    result = dict(id=key, tool=tool, args=args, depends_on=list(dependencies), allow_failed_dependencies=allow_failed)
    if resources is not None:
        result['resources'] = resources
    return result


def completion_nodes(dependencies: list[str]):
    return [node('review', 'review', {'evidence': {'$results': dependencies}}, dependencies, True),
            node('package', 'package', {'evidence': [ref('review')]}, ['review'])]


def initial_plan(request: dict) -> list[dict]:
    mode, settings, inputs = request['mode'], request['settings'], request['inputs']
    if mode == 'custom':
        return request['nodes']
    if mode == 'resolve':
        from .resolve import validate_settings

        settings = validate_settings(settings)
        if 'hkl' not in inputs:
            raise ValueError('resolve requires a typed reflection input named hkl')
        controls = [key for key in inputs if key == 'ins' or key.startswith('laue_')]
        if len(controls) > 8:
            raise ValueError('At most eight predeclared Laue-group controls')
        if settings.get('solver') == 'shelxt' and not controls:
            raise ValueError('SHELXT initialization requires a paired control file')
        prepare = dict(hkl=input_ref('hkl'), settings=settings)
        if 'structure' in inputs:
            prepare['structure'] = input_ref('structure')
        if controls:
            prepare['ins'] = input_ref('ins' if 'ins' in controls else controls[0])
        plan = [node('evidence', 'resolve.prepare', prepare)]
        seeds = []
        if 'structure' in inputs or not controls or settings.get('solver') != 'shelxt':
            seeds.append(node('seed', 'resolve.seed', {'preparation': ref('evidence')}, ['evidence']))
        if settings.get('solver') == 'shelxt':
            for i, control in enumerate(controls):
                seeds.append(node(f'shelxt-{i}', 'resolve.seed', {'preparation': ref('evidence'), 'ins': input_ref(control)},
                                  ['evidence']))
        seed_ids = [n['id'] for n in seeds]
        plan.extend(seeds)
        plan.append(node('screen', 'resolve.screen',
                         {'preparation': ref('evidence'), 'seeds': {'$results': seed_ids}},
                         ['evidence', *seed_ids], True))
        probe = settings.get('pixel_probe')
        if probe is not None:
            roles = ('experiments', 'predictions', 'profiles')
            if not all(role in inputs for role in roles):
                raise ValueError('pixel_probe requires experiments, predictions and profiles inputs')
            plan.append(node('targeted-pixels', 'analysis.run',
                             {'analysis': 'build_overlaps', 'inputs': [input_ref(role) for role in roles],
                              'attachments': [input_ref(key) for key in sorted(inputs) if key.startswith('template_')],
                              'options': probe}, ['evidence']))
            plan.append(node('pixel-fit', 'analysis.run',
                             {'analysis': 'joint_fit', 'inputs': [ref('targeted-pixels', '/manifest')]},
                             ['targeted-pixels']))
            corrections = [role for role in ('corrections', 'scale_geometry') if role in inputs]
            if len(corrections) > 1:
                raise ValueError('Supply corrections or a global-scale geometry recipe for the selected groups')
            if settings.get('pixel_export_hklf5') and not corrections:
                raise ValueError('HKLF 5 export requires corrected observation groups with same-phase evidence')
            if corrections:
                role = corrections[0]
                plan.append(node('pixel-correct', 'analysis.run',
                                 {'analysis': 'correct_groups' if role == 'corrections' else 'global_scale',
                                  'inputs': [ref('pixel-fit', '/groups'), input_ref(role)]}, ['pixel-fit']))
            if settings.get('pixel_export_hklf5'):
                plan.append(node('pixel-export', 'analysis.run',
                                 {'analysis': 'export_hklf5', 'inputs': [ref('pixel-correct', '/corrected_groups')]},
                                 ['pixel-correct']))
        return plan
    required = {'audit': ['cif'], 'solve': ['ins', 'hkl'], 'refine': ['res', 'hkl'],
                'feedback': ['observations'], 'reduce': []}[mode]
    if not all(x in inputs for x in required):
        raise ValueError('mode requires inputs: '+', '.join(required))
    if mode == 'audit':
        return [node('audit', 'crystal.audit', dict(cif=input_ref('cif'),
                     expected_elements=settings.get('expected_elements'))), *completion_nodes(['audit'])]
    if mode == 'solve':
        return [node('solve', 'structure.solve', dict(ins=input_ref('ins'), hkl=input_ref('hkl')))]
    if mode == 'refine':
        return [node('refine', 'structure.refine', dict(res=input_ref('res'), hkl=input_ref('hkl'),
                    cycles=settings.get('cycles', 8))), *completion_nodes(['refine'])]
    if mode == 'feedback':
        models = []
        for value in settings.get('models', []):
            model = {**value, 'cifs': [input_ref(x) for x in value['cifs']]}
            models.append(model)
        if not models:
            raise ValueError('feedback mode requires settings.models with declared provenance')
        return [node('feedback', 'feedback.loop', dict(observations=input_ref('observations'), models=models,
                     seed=settings.get('seed', 20261030), absolute_sd=settings.get('absolute_sd', 1.),
                     relative_sd=settings.get('relative_sd', .2))), *completion_nodes(['feedback'])]
    if not inputs:
        raise ValueError('reduce mode requires image files')
    if not settings.get('composition'):
        raise ValueError('reduce mode requires the chemical composition')
    configured = settings.get('dials_options', {})
    spot_workers = min(4, request['scheduler']['cpu_threads'])
    plan = [node('import', 'dials.run', dict(operation='import', inputs=[input_ref(x) for x in sorted(inputs)],
                                           options=configured.get('import', {}))),
            node('find_spots', 'dials.run', dict(operation='find_spots', inputs=[ref('import', '/experiments')],
                                                options=configured.get('find_spots', {})), ['import'],
                 resources={'cpu_threads': spot_workers}),
            node('index', 'dials.run', dict(operation='index', inputs=[ref('import', '/experiments'), ref('find_spots', '/reflections')],
                                           options=configured.get('index', {})), ['import', 'find_spots'])]
    previous = 'index'
    for key, operation in [('geometry', 'refine'), ('integrate', 'integrate'), ('scale', 'scale'), ('export', 'export')]:
        options = dict(configured.get(operation, {}))
        if operation == 'export':
            options['shelx.composition'] = settings['composition']
            options.setdefault('intensity', 'scale')
        plan.append(node(key, 'dials.run', dict(operation=operation, inputs=[ref(previous, '/experiments'), ref(previous, '/reflections')],
                                               options=options), [previous]))
        previous = key
    plan.append(node('solve', 'structure.solve', dict(ins=ref('export', '/ins'), hkl=ref('export', '/hkl')), ['export']))
    return plan


def _validate_input_refs(value, inputs):
    if isinstance(value, dict):
        if '$input' in value and value['$input'] not in inputs:
            raise ValueError('unknown input reference: '+value['$input'])
        for child in value.values():
            _validate_input_refs(child, inputs)
    elif isinstance(value, list):
        for child in value:
            _validate_input_refs(child, inputs)


def initialize(request: dict, directory: str | Path, *, base: str | Path = '.',
               tools=TOOLS, worker_module=DEFAULT_ACTION_WORKER) -> dict:
    request = validate_request(request, tools=tools)
    root = Path(directory).expanduser().resolve()
    if root.exists():
        raise FileExistsError('session directory must not exist; choose a new directory')
    plan = initial_plan(request)
    for item in plan:
        _validate_input_refs(item['args'], request['inputs'])
        task_resources(item, request['scheduler'])
    # Validate input existence before creating the session.
    sources = {}
    for key, raw in request['inputs'].items():
        path = Path(raw).expanduser()
        path = (Path(base)/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.name.startswith('-') or '=' in path.name:
            raise ValueError('input basename could be interpreted as a native command option')
        sources[key] = path
    root.mkdir(parents=True)
    store = Store(root, create=True)
    try:
        manifest = {}
        image_root = Path(os.path.commonpath([str(p.parent) for p in sources.values()])) if request['mode'] == 'reduce' else None
        for key, source in sources.items():
            # A rotation sequence must keep its shared directory/template.
            # Per-key directories turn a scan into unrelated one-frame imagesets.
            target = (root/'inputs'/'images'/source.relative_to(image_root) if image_root is not None
                      else root/'inputs'/key/source.name)
            manifest[key] = snapshot_input(root, source, target)
        store.set('request', request)
        store.set('request_digest', store.digest(request))
        store.set('inputs', manifest)
        store.set('build', BUILD)
        if worker_module != DEFAULT_ACTION_WORKER:
            store.set('action_worker_module', worker_module)
        import importlib.metadata, sys
        versions = {}
        for package in ('numpy', 'scipy', 'gemmi', 'mcp', 'dials', 'cctbx-base'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        atomic_json(root/'runtime.json', {'build': BUILD, 'python': sys.version, 'dependencies': versions})
        store.event('session_initialized', dict(build=BUILD, request_digest=store.digest(request), inputs=manifest))
        store.add_nodes(plan, tools=tools)
        atomic_json(root/'request.json', request)
        atomic_json(root/'inputs_manifest.json', manifest)
        return store.snapshot()
    except BaseException:
        store.set('initialization_failed', True)
        raise
    finally:
        store.close()


def resolve(value: Any, store: Store, dependencies: list[str]) -> Any:
    if isinstance(value, dict):
        if '$input' in value:
            item = store.get('inputs')[value['$input']]
            artifact_path(store.root, item)
            return item
        if '$ref' in value:
            spec = value['$ref']
            if spec['node'] not in dependencies:
                raise ValueError('reference is not a direct dependency')
            producer = store.node(spec['node'])
            if producer['status'] != 'succeeded':
                raise ValueError('referenced producer did not succeed')
            return pointer(producer['result'], spec.get('pointer', ''))
        if '$results' in value:
            results = []
            for key in value['$results']:
                if key not in dependencies:
                    raise ValueError('result source is not a direct dependency')
                producer = store.node(key)
                results.append(producer['result'] if producer['status'] == 'succeeded' else
                               dict(execution_failed=True, node=key, status=producer['status'], error=producer['error']))
            return results
        return {key: resolve(child, store, dependencies) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve(child, store, dependencies) for child in value]
    return value


def _expand_solve(store: Store):
    request = store.get('request')
    if request['mode'] not in {'solve', 'reduce'}:
        return
    solve = store.node('solve')
    if solve['status'] != 'succeeded' or solve['expanded']:
        return
    candidates = solve['result']['candidates']
    if len(candidates) > 32:
        raise NeedsReview('more than 32 candidates; explicit selection is required')
    planned, ids = [], []
    for i, candidate in enumerate(candidates):
        if not candidate['eligible_for_refinement']:
            continue
        key = f'refine_{i:02d}'
        planned.append(node(key, 'structure.refine', dict(res=ref('solve', f'/candidates/{i}/res'),
                            hkl=ref('solve', f'/candidates/{i}/hkl'), cycles=request['settings'].get('cycles', 8)), ['solve']))
        ids.append(key)
    if ids:
        planned.append(node('compare', 'candidates.compare', {'candidates': {'$results': ids}}, ids, True))
        planned += completion_nodes(['compare'])
    else:
        planned += completion_nodes(['solve'])
    _install_expansion(store, 'solve', planned, 'candidate_graph_expanded',
                       dict(actual_candidates=len(candidates), refinement_nodes=ids))


def _install_expansion(store, producer, planned, event, details):
    present = {x['id'] for x in store.summaries()}
    for intended in planned:
        if intended['id'] in present:
            current = store.node(intended['id'])
            if (current['tool'] != intended['tool'] or current['args'] != intended['args']
                    or current['dependencies'] != intended['depends_on']):
                raise ValueError('candidate expansion conflicts with an existing node definition')
    remaining = [x for x in planned if x['id'] not in present]
    if remaining:
        store.add_nodes(remaining)
    store.db.execute('UPDATE nodes SET expanded=1 WHERE id=?', (producer,))
    store.event(event, details)


def _expand_candidates(store):
    _expand_solve(store)
    request = store.get('request')
    if request['mode'] != 'resolve':
        return
    definitions = {n['id']: n for n in store.summaries()}
    for producer, field, stage, next_id in (('screen', 'shortlist', 'short', 'promote'),
                                            ('promote', 'finalists', 'full', 'select')):
        if definitions.get(producer, {}).get('status') != 'succeeded':
            continue
        upstream = store.node(producer)
        if upstream['expanded']:
            continue
        values = upstream['result'][field]
        ids = [f"{stage}-{value['id']}" for value in values]
        planned = [node(key, 'resolve.fit', {'preparation': ref('evidence'),
                        'candidate': ref(producer, f'/{field}/{i}'), 'stage': stage},
                        ['evidence', producer]) for i, key in enumerate(ids)]
        arguments = {'preparation': ref('evidence'), 'fits': {'$results': ids}}
        upstream_ids = ['evidence', producer, *ids]
        if stage == 'full':
            arguments['screen'] = ref('screen')
            upstream_ids.append('screen')
        planned.append(node(next_id, f'resolve.{next_id}', arguments, upstream_ids, True))
        if stage == 'full':
            dependencies = ['screen', 'promote', 'select', 'final-audit',
                            *(key for key in definitions if key.startswith('shelxt-'))]
            dependencies.extend(key for key in ('pixel-fit', 'pixel-correct', 'pixel-export') if key in definitions)
            planned.append(node('final-audit', 'resolve.audit',
                                {'preparation': ref('evidence'), 'selection': ref('select')}, ['evidence', 'select']))
            if request['settings'].get('crosscheck'):
                checks = {'selection': ref('select')}
                for role in ('fcf', 'fcf_control'):
                    if role in request['inputs']:
                        checks[role] = input_ref(role)
                planned.append(node('crosscheck', 'resolve.crosscheck', checks, ['select']))
                dependencies.append('crosscheck')
            planned.extend(completion_nodes(dependencies))
        _install_expansion(store, producer, planned, 'resolution_candidates_expanded',
                           {'stage': stage, 'candidates': ids})


def capabilities() -> dict:
    commands = ['shelxt', 'shelxl', 'platon'] + ['dials.'+op for op in DIALS_OPTIONS]
    return dict(product='Ariadne', version='0.8.0', build=BUILD,
                modes=['audit', 'solve', 'refine', 'reduce', 'feedback', 'resolve', 'custom'],
                skills=['ariadne-reduction', 'solve-single-crystal'],
                actions={name: dict(required=sorted(spec.required), optional=sorted(spec.optional), description=spec.description)
                         for name, spec in TOOLS.items()},
                executable_availability={name: shutil.which(name) for name in commands},
                python_modules={name: importlib.util.find_spec(name) is not None for name in ['numpy', 'scipy', 'gemmi', 'dials', 'cctbx', 'mcp']},
                analysis_wrappers=sorted(ANALYSES),
                automatic_raw_route='single crystal; complex twins require explicit joint-analysis workflow',
                scientific_acceptance='not_established_by_execution')


def _experts(request):
    try:
        root = plugin_root()/'experts'
    except RuntimeError:
        return []
    result = []
    for name in request['experts']:
        check_name(name)
        path = root/(name+'.json')
        if not path.is_file():
            raise ValueError('expert pack not found: '+name)
        pack = read_json(path)
        if pack.get('schema') != 'ariadne-expert/1':
            raise ValueError('unsupported expert pack')
        result.append(pack)
    return result


def _verify(store: Store):
    if store.get('initialization_failed'):
        raise RuntimeError('session initialization did not complete')
    if store.digest(store.get('request')) != store.get('request_digest') or not store.verify_events():
        raise RuntimeError('state/event integrity check failed')
    import json
    definitions, finished = {}, {}
    for event in store.db.execute("SELECT kind,payload FROM events WHERE kind IN ('node_added','attempt_finished') ORDER BY seq"):
        payload = json.loads(event['payload'])
        if event['kind'] == 'node_added':
            definitions[payload['id']] = payload
        else:
            finished[payload['node']] = payload
    for item in store.get('inputs').values():
        artifact_path(store.root, item)
    for summary in store.summaries():
        current = store.node(summary['id'])
        definition = dict(id=current['id'], tool=current['tool'], args=current['args'],
                          depends_on=current['dependencies'], allow_failed_dependencies=bool(current['allow_failed']))
        if current['resources'] or 'resources' in definitions.get(summary['id'], {}):
            definition['resources'] = current['resources']
        if definition != definitions.get(summary['id']):
            raise RuntimeError('stored node definition does not match its event history')
        if summary['status'] == 'succeeded':
            actual = current['result']
            if store.digest(actual) != finished.get(summary['id'], {}).get('result_digest'):
                raise RuntimeError('stored result does not match its event history')
            payload = dict(actual)
            record = payload.pop('result_artifact', None)
            if record is None or dumps(read_json(artifact_path(store.root, record))) != dumps(payload):
                raise RuntimeError('stored result does not match its scientific result artifact')
            for item in collect_artifacts(actual):
                artifact_path(store.root, item)


def run(directory: str | Path, *, max_steps: int | None = None, force_isolated: bool = False,
        worker_module=DEFAULT_ACTION_WORKER) -> dict:
    root = Path(directory).expanduser().resolve()
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError('max_steps must be a positive integer')
    with exclusive_lock(root/'.runner.lock'):
        store = Store(root)
        clock_started = False
        try:
            if store.get('action_worker_module', DEFAULT_ACTION_WORKER) != worker_module:
                raise ValueError('This session requires its application executor')
            _verify(store)
            # An interrupted native process may still exist. Never silently rerun it.
            running = [x['id'] for x in store.summaries() if x['status'] == 'running']
            if running:
                raise RuntimeError('unsettled previous attempts: '+', '.join(running)+'. Inspect logs and use recover explicitly.')
            store.start_clock()
            clock_started = True
            configuration = validate_scheduler(store.get('request').get('scheduler'))
            from .host_budget import HostBudget
            host = HostBudget.configured()
            if (configuration['isolate_actions'] or force_isolated or host is not None
                    or worker_module != DEFAULT_ACTION_WORKER):
                from .scheduler import run_isolated
                run_isolated(store, configuration, max_steps=max_steps, choose=choose, resolve=resolve,
                             expand=_expand_candidates, capabilities={'actions': capabilities()['actions']},
                             experts=_experts(store.get('request')), validate_inputs=_validate_input_refs,
                             host=host, worker_module=worker_module)
                store.stop_clock()
                clock_started = False
                result = store.snapshot()
                atomic_json(root/'status.json', result)
                return result
            steps = 0
            while True:
                _expand_candidates(store)
                store.propagate_failures()
                request = store.get('request')
                budget = request['budget']
                if store.get('cancel_requested'):
                    break
                if (max_steps is not None and steps >= max_steps) or store.wall_elapsed() >= budget['wall_seconds']:
                    break
                if store.get('actions_used') >= budget['max_actions'] or store.get('decisions_used') >= budget['max_decisions']:
                    break
                observation = store.snapshot()
                if not observation['ready']:
                    break
                planning_started = time.monotonic()
                planner_config = dict(request['planner'])
                planner_config['timeout_seconds'] = min(planner_config.get('timeout_seconds', 45),
                                                        budget['wall_seconds']-store.wall_elapsed())
                try:
                    decision = choose(planner_config, observation,
                                      {'actions': capabilities()['actions']}, _experts(request))
                except Exception as exc:
                    store.event('planner_failed', dict(error=str(exc)))
                    raise
                finally:
                    store.set('planning_seconds', store.get('planning_seconds', 0.)+time.monotonic()-planning_started)
                store.set('decisions_used', store.get('decisions_used')+1)
                store.event('planner_decision', decision)
                if decision['action'] == 'pause':
                    break
                if decision['action'] == 'propose':
                    proposed = decision['node']
                    _validate_input_refs(proposed.get('args', {}), store.get('inputs'))
                    if len(store.summaries()) >= budget['max_actions']:
                        raise RuntimeError('proposed graph exceeds action budget')
                    store.add_nodes([proposed])
                    continue
                key = decision.get('node')
                if key not in observation['ready']:
                    raise ValueError('planner selected a node that is not ready')
                n = store.node(key)
                work = store.begin(key)
                started = time.monotonic()
                try:
                    args = resolve(n['args'], store, n['dependencies'])
                    context = Context(store, key, work, started, resources=task_resources(n, configuration))
                    if context.remaining() <= 0:
                        raise TimeoutError('time budget exhausted before scientific action')
                    from threadpoolctl import threadpool_limits
                    with threadpool_limits(limits=context.resources['cpu_threads']):
                        result = execute(context, n['tool'], args)
                    if context.remaining() <= 0:
                        raise TimeoutError('scientific action exceeded its time budget')
                    if store.get('cancel_requested'):
                        raise Cancelled('cancelled before committing calculation result')
                    store.finish(key, 'succeeded', time.monotonic()-started, result=result)
                except Cancelled as exc:
                    store.finish(key, 'cancelled', time.monotonic()-started, error=str(exc))
                except NeedsReview as exc:
                    store.finish(key, 'blocked', time.monotonic()-started, error=str(exc))
                except KeyboardInterrupt:
                    store.finish(key, 'interrupted', time.monotonic()-started, error='user interrupt')
                    raise
                except Exception as exc:
                    store.finish(key, 'failed', time.monotonic()-started, error=f'{type(exc).__name__}: {exc}')
                steps += 1
            _expand_candidates(store)
            store.propagate_failures()
            store.stop_clock()
            clock_started = False
            result = store.snapshot()
            atomic_json(root/'status.json', result)
            return result
        finally:
            if clock_started:
                store.stop_clock()
            store.close()


def status(directory):
    store = Store(Path(directory))
    try:
        return store.snapshot()
    finally:
        store.close()


def cancel(directory):
    store = Store(Path(directory))
    try:
        store.set('cancel_requested', True)
        store.event('cancel_requested', {})
        return {'cancel_requested': True}
    finally:
        store.close()


def resume(directory):
    root = Path(directory).resolve()
    with exclusive_lock(root/'.runner.lock'):
        store = Store(root)
        try:
            store.set('cancel_requested', False)
            store.event('cancel_cleared', {})
            return store.snapshot()
        finally:
            store.close()


def retry(directory, node_id):
    root = Path(directory).resolve()
    with exclusive_lock(root/'.runner.lock'):
        store = Store(root)
        try:
            n = store.node(node_id)
            if n['status'] not in {'failed', 'cancelled', 'blocked', 'interrupted'}:
                raise ValueError('only a terminal unsuccessful node can be retried')
            if n['attempts'] >= store.get('request')['budget']['max_attempts']:
                raise RuntimeError('attempt budget exhausted')
            if n['tool'] == 'feedback.loop':
                raise RuntimeError('feedback test exposure may already have occurred; create an explicit new experiment, not an automatic retry')
            rows = list(store.db.execute('SELECT id,dependencies,status FROM nodes'))
            descendants, frontier = set(), {node_id}
            while frontier:
                following = {r['id'] for r in rows if set(__import__('json').loads(r['dependencies'])) & frontier} - descendants
                descendants.update(following)
                frontier = following
            consumers = [r['id'] for r in rows if r['id'] in descendants and r['status'] not in {'pending', 'skipped'}]
            if consumers:
                raise ValueError('downstream nodes already consumed this result; create a new branch/session')
            store.db.execute('UPDATE nodes SET status="pending",error=NULL,result=NULL WHERE id=?', (node_id,))
            for key in descendants:
                store.db.execute('UPDATE nodes SET status="pending",error=NULL WHERE id=? AND status="skipped"', (key,))
            store.event('retry_authorized', dict(node=node_id))
            return store.snapshot()
        finally:
            store.close()


def recover(directory, node_id, *, confirmed_process_stopped: bool = False):
    if not confirmed_process_stopped:
        raise ValueError('inspect the process and explicitly confirm it has stopped before recovery')
    root = Path(directory).resolve()
    with exclusive_lock(root/'.runner.lock'):
        store = Store(root)
        try:
            n = store.node(node_id)
            if n['status'] != 'running':
                raise ValueError('node is not an unsettled running attempt')
            # Do not assume PID ownership after a crash; caller confirmation is explicit.
            from datetime import datetime, timezone
            started = store.db.execute('SELECT started FROM attempts WHERE node=? AND number=?', (node_id, n['attempts'])).fetchone()[0]
            conservative_elapsed = max(0., datetime.now(timezone.utc).timestamp()-datetime.fromisoformat(started).timestamp())
            store.finish(node_id, 'interrupted', conservative_elapsed, error='operator confirmed previous process stopped; elapsed time conservatively includes downtime')
            if not any(item['status'] == 'running' for item in store.summaries()):
                store.stop_clock()
            return store.snapshot()
        finally:
            store.close()


def export_session(directory, destination, *, include_inputs=False):
    root, target = Path(directory).resolve(), Path(destination).resolve()
    if target.exists() or target.is_relative_to(root):
        raise ValueError('export path must be new and outside the session')
    with exclusive_lock(root/'.runner.lock'):
        store = Store(root)
        try:
            _verify(store)
            if any(x['status'] == 'running' for x in store.summaries()):
                raise RuntimeError('cannot export an unsettled running session')
            files = [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                     and p.name not in {'agent.sqlite', 'agent.sqlite-wal', 'agent.sqlite-shm', '.runner.lock'}
                     and (include_inputs or not p.is_relative_to(root/'inputs'))]
            evidence = dict(build=BUILD, request=store.get('request'), status=store.snapshot(),
                            nodes=[store.node(x['id']) for x in store.summaries()],
                            events=[dict(r) for r in store.db.execute('SELECT * FROM events ORDER BY seq')])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.tmp')
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('session.json', dumps(evidence)+'\n')
                archive.writestr('file_manifest.json', dumps([
                    dict(path=p.relative_to(root).as_posix(), content_id=content_id(p), size=p.stat().st_size) for p in files])+'\n')
                for path in files:
                    archive.write(path, path.relative_to(root).as_posix())
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError('export ZIP verification failed')
            os.replace(temporary, target)
            return dict(path=str(target), content_id=content_id(target), size_bytes=target.stat().st_size,
                        includes_inputs=include_inputs)
        finally:
            store.close()
