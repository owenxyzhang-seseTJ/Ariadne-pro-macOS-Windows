"""Small explicit schemas, independent of MCP or a language-model SDK."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{0,79}$')
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted', 'blocked', 'skipped'}


def validate_scheduler(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value)-{'max_workers', 'cpu_threads', 'memory_mb',
                                                 'default_task_memory_mb', 'isolate_actions'}:
        raise ValueError('unknown scheduler configuration')
    workers = value.get('max_workers', 1)
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('scheduler max_workers must be an integer in 1..64')
    result = {'max_workers': 1, 'cpu_threads': workers, 'memory_mb': None,
              'default_task_memory_mb': 256, 'isolate_actions': workers > 1}
    result.update(value)
    for key, upper in (('max_workers', 64), ('cpu_threads', 4096), ('default_task_memory_mb', 16_777_216)):
        if type(result[key]) is not int or not 1 <= result[key] <= upper:
            raise ValueError(f'scheduler {key} must be an integer in 1..{upper}')
    if result['memory_mb'] is not None and (type(result['memory_mb']) is not int or not 1 <= result['memory_mb'] <= 16_777_216):
        raise ValueError('scheduler memory_mb must be a positive integer or null')
    if type(result['isolate_actions']) is not bool or result['max_workers'] > 1 and not result['isolate_actions']:
        raise ValueError('concurrent actions require isolate_actions=true')
    return result


def task_resources(node, scheduler):
    """Declared admission tokens; memory is not a measured or enforced RSS."""
    resource = {'cpu_threads': 1, 'memory_mb': scheduler['default_task_memory_mb']}
    resource.update(node.get('resources') or {})
    if resource['cpu_threads'] > scheduler['cpu_threads']:
        raise ValueError('node CPU request exceeds the session cpu_threads budget')
    if scheduler['memory_mb'] is not None and resource['memory_mb'] > scheduler['memory_mb']:
        raise ValueError('node memory reservation exceeds the session memory_mb budget')
    return resource


@dataclass(frozen=True)
class ToolSpec:
    required: frozenset[str]
    optional: frozenset[str]
    description: str


TOOLS = {
    'artifact.inspect': ToolSpec(frozenset({'artifact'}), frozenset({'json_pointer', 'max_chars'}), 'Read bounded text or a selected JSON value from a verified artifact'),
    'files.unpack': ToolSpec(frozenset({'archive'}), frozenset({'extensions', 'max_bytes', 'max_files'}), 'Stage a ZIP input bundle with sidecars preserved'),
    'crystal.audit': ToolSpec(frozenset({'cif'}), frozenset({'block', 'expected_elements'}), 'Read and validate a real CIF'),
    'crystal.predict': ToolSpec(frozenset({'cifs', 'reflections'}), frozenset({'blocks'}), 'Native X-ray structure factors'),
    'feedback.condition': ToolSpec(frozenset({'problem'}), frozenset(), 'Signed Gaussian structure-assisted posterior'),
    'feedback.loop': ToolSpec(frozenset({'observations', 'models'}), frozenset({'seed', 'absolute_sd', 'relative_sd'}), 'Training/validation/frozen final test'),
    'dials.run': ToolSpec(frozenset({'operation', 'inputs'}), frozenset({'options'}), 'Fixed DIALS command with typed PHIL'),
    'analysis.run': ToolSpec(frozenset({'analysis', 'inputs'}), frozenset({'options'}), 'Run a registered R2 scientific analysis'),
    'structure.solve': ToolSpec(frozenset({'ins', 'hkl'}), frozenset(), 'Run SHELXT and discover actual paired candidate files'),
    'structure.refine': ToolSpec(frozenset({'res', 'hkl'}), frozenset({'cycles'}), 'Run SHELXL with paired data, then audit CIF'),
    'native.run': ToolSpec(frozenset({'request'}), frozenset({'inputs'}), 'Run typed native integrate, solve-observations, solve, autosolve, symmetry or joint-refine in a bounded isolated process'),
    'candidates.compare': ToolSpec(frozenset({'candidates'}), frozenset(), 'Descriptive comparison; not automatic space-group acceptance'),
    'review': ToolSpec(frozenset({'evidence'}), frozenset(), 'Summarize actual preceding results'),
    'package': ToolSpec(frozenset({'evidence'}), frozenset(), 'Archive referenced results and scientific artifacts'),
}


def check_name(value: Any) -> str:
    if not isinstance(value, str) or NAME.fullmatch(value) is None:
        raise ValueError('invalid identifier')
    return value


def references(value: Any) -> set[str]:
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            raise ValueError('literal artifact claims are forbidden; use $input or $ref')
        if '$ref' in value:
            if set(value) != {'$ref'} or not isinstance(value['$ref'], dict):
                raise ValueError('invalid reference')
            ref = value['$ref']
            if set(ref)-{'node', 'pointer'} or 'node' not in ref or not isinstance(ref.get('pointer', ''), str):
                raise ValueError('invalid reference fields')
            return {check_name(ref['node'])}
        if '$results' in value:
            if set(value) != {'$results'} or not isinstance(value['$results'], list):
                raise ValueError('invalid result collection')
            return {check_name(x) for x in value['$results']}
        if '$input' in value:
            if set(value) != {'$input'}:
                raise ValueError('invalid input reference')
            check_name(value['$input'])
            return set()
        return set().union(*(references(x) for x in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(references(x) for x in value)) if value else set()
    return set()


def validate_node(node: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError('node must be an object')
    if set(node)-{'id', 'tool', 'args', 'depends_on', 'allow_failed_dependencies', 'resources'}:
        raise ValueError('unknown node fields')
    check_name(node.get('id'))
    if node.get('tool') not in TOOLS:
        raise ValueError('unregistered tool')
    args = node.get('args', {})
    if not isinstance(args, dict):
        raise ValueError('tool arguments must be an object')
    spec = TOOLS[node['tool']]
    if not spec.required <= args.keys() or set(args)-spec.required-spec.optional:
        raise ValueError('missing or unknown tool arguments')
    dependencies = node.get('depends_on', [])
    if not isinstance(dependencies, list) or len(set(dependencies)) != len(dependencies):
        raise ValueError('dependencies must be a list of unique node IDs')
    for dependency in dependencies:
        check_name(dependency)
        if dependency == node['id']:
            raise ValueError('node depends on itself')
    refs = references(args)
    if not refs <= set(dependencies):
        raise ValueError('references must be declared direct dependencies')
    if node['tool'] in {'candidates.compare', 'review', 'package'} and not refs:
        raise ValueError('review and packaging require real producer references')
    if node['tool'] in {'candidates.compare', 'review', 'package'}:
        evidence = args['candidates'] if node['tool'] == 'candidates.compare' else args['evidence']
        if not (isinstance(evidence, dict) and set(evidence) == {'$results'} or isinstance(evidence, list) and evidence and all(isinstance(x, dict) and set(x) == {'$ref'} for x in evidence)):
            raise ValueError('evidence aggregation accepts producer references only')
    failed = node.get('allow_failed_dependencies', False)
    if not isinstance(failed, bool) or failed and node['tool'] not in {'candidates.compare', 'review', 'package'}:
        raise ValueError('only evidence aggregation may consume failed dependencies')
    output = dict(id=node['id'], tool=node['tool'], args=args, depends_on=dependencies,
                  allow_failed_dependencies=failed)
    if 'resources' in node:
        resources = node['resources']
        if not isinstance(resources, dict) or set(resources)-{'cpu_threads', 'memory_mb'}:
            raise ValueError('unknown node resources')
        for key, value in resources.items():
            maximum = 4096 if key == 'cpu_threads' else 16_777_216
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('node resources must be bounded positive integers')
        output['resources'] = dict(resources)
    return output


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError('request must be an object')
    allowed = {'schema', 'mode', 'inputs', 'nodes', 'budget', 'executables', 'planner', 'settings', 'experts', 'scheduler'}
    if set(request)-allowed:
        raise ValueError('unknown request fields')
    if request.get('schema', 'ariadne-agent/1') != 'ariadne-agent/1':
        raise ValueError('unsupported request schema')
    mode = request.get('mode')
    if mode not in {'audit', 'solve', 'refine', 'reduce', 'feedback', 'custom'}:
        raise ValueError('unsupported mode')
    inputs = request.get('inputs', {})
    if not isinstance(inputs, dict):
        raise ValueError('inputs must be a name-to-path mapping')
    for key, value in inputs.items():
        check_name(key)
        if not isinstance(value, str) or not value:
            raise ValueError('input paths must be nonempty strings')
    budget = dict(max_actions=100, max_attempts=2, wall_seconds=7200., action_seconds=600., max_decisions=200)
    if not isinstance(request.get('budget', {}), dict):
        raise ValueError('budget must be an object')
    if set(request.get('budget', {}))-set(budget):
        raise ValueError('unknown budget field')
    budget.update(request.get('budget', {}))
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 864000:
            raise ValueError('invalid positive budget')
        if key in {'max_actions', 'max_attempts', 'max_decisions'} and not isinstance(value, int):
            raise ValueError('count budgets must be integers')
    executable_names = {'shelxt', 'shelxl', 'scientific_python'} | {'dials.'+x for x in
                        ['import', 'find_spots', 'index', 'refine', 'integrate', 'scale', 'export', 'predict']}
    executable_config = request.get('executables', {})
    if not isinstance(executable_config, dict) or set(executable_config)-executable_names:
        raise ValueError('unknown executable mapping')
    if any(not isinstance(v, str) or not v or '\x00' in v for v in executable_config.values()):
        raise ValueError('executable values are file paths, not shell commands')
    settings = request.get('settings', {})
    if not isinstance(settings, dict) or set(settings)-{'cycles', 'composition', 'dials_options', 'expected_elements', 'seed', 'models', 'absolute_sd', 'relative_sd'}:
        raise ValueError('unknown workflow setting')
    planner = request.get('planner', {'kind': 'deterministic'})
    if not isinstance(planner, dict) or planner.get('kind') not in {'deterministic', 'chat-completions'}:
        raise ValueError('unknown planner')
    if set(planner)-{'kind', 'base_url', 'model', 'api_key_env', 'allow_local_http', 'timeout_seconds'}:
        raise ValueError('unknown planner field or literal secret; use api_key_env')
    timeout = planner.get('timeout_seconds', 45)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
        raise ValueError('planner timeout_seconds must be in (0,600]')
    experts = request.get('experts', ['crystallography'])
    if not isinstance(experts, list) or any(not isinstance(x, str) for x in experts):
        raise ValueError('experts must be a list of names')
    if not isinstance(request.get('nodes', []), list):
        raise ValueError('nodes must be a list')
    nodes = [validate_node(x) for x in request.get('nodes', [])]
    scheduler = validate_scheduler(request.get('scheduler'))
    for item in nodes:
        task_resources(item, scheduler)
    if mode == 'custom' and not nodes:
        raise ValueError('custom mode requires nodes')
    if mode != 'custom' and nodes:
        raise ValueError('explicit nodes are only accepted in custom mode')
    return {**request, 'schema': 'ariadne-agent/1', 'budget': budget, 'inputs': inputs,
            'executables': executable_config, 'settings': settings, 'planner': planner, 'experts': experts,
            'nodes': nodes, 'scheduler': scheduler}
