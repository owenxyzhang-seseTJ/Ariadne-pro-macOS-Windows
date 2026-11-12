"""Install both release wheels into a fresh target and run an external smoke test.

Run with the project's validation environment:
    python scripts/check_v08_wheels.py

The target contains only freshly installed Ariadne wheels. Third-party
dependencies are deliberately reused from the invoking environment. This
checks wheel integration, console entry points and actual import locations;
it is not a clean-machine dependency installation test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# This code runs in a fresh interpreter outside the checkout. In particular,
# editable-install metadata is not accepted as evidence of a wheel import.
PROBE = r'''
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

target = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
assert not Path.cwd().resolve().is_relative_to(source), 'Probe is inside source checkout'
names = ['ariadne', 'ariadne.agent.engine', 'ariadne.agent.scheduler',
         'ariadne.agent.action_worker', 'ariadne_xtal', 'ariadne_xtal.agent',
         'ariadne_xtal.refinement', 'ariadne_xtal.kernels']
modules = {}
for name in names:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(target), f'{name} was shadowed by {path}'
    modules[name] = str(path)
from ariadne_xtal.kernels import kernel_status
kernel = kernel_status()
if kernel['compiled_available']:
    native = importlib.import_module('ariadne_xtal._intensity_cpp')
    path = Path(native.__file__).resolve()
    assert path.is_relative_to(target), f'C++ extension came from {path}'
    modules['ariadne_xtal._intensity_cpp'] = str(path)

distributions = {}
for distribution in metadata.distributions(path=[str(target)]):
    name = canonicalize_name(distribution.metadata['Name'])
    if name not in {'ariadne', 'ariadne-xtal'}:
        continue
    assert distribution.version == '0.8.0', (name, distribution.version)
    python_requirement = distribution.metadata.get('Requires-Python')
    assert not python_requirement or platform.python_version() in SpecifierSet(python_requirement)
    requirements = []
    for text in distribution.requires or []:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate({'extra': ''}):
            continue
        installed = metadata.version(requirement.name)
        assert installed in requirement.specifier, (text, installed)
        requirements.append({'requirement': text, 'installed': installed, 'satisfied': True})
    distributions[name] = {'version': distribution.version, 'requires_python': python_requirement,
                           'requirements': requirements}
assert set(distributions) == {'ariadne', 'ariadne-xtal'}, distributions
from ariadne.agent.backends import plugin_root
assert plugin_root() == source
print(json.dumps({'working_directory': str(Path.cwd()), 'target': str(target),
                  'module_files': modules, 'distributions': distributions,
                  'all_ariadne_modules_from_target': True, 'kernel': kernel,
                  'legacy_resource_root': str(plugin_root())}, indent=2))
'''


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def choose_native_wheel(directory):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    supported = set(sys_tags())
    candidates = []
    for path in directory.glob('ariadne_xtal-0.8.0-*.whl'):
        _, _, _, tags = parse_wheel_filename(path.name)
        if tags & supported:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError('No native wheel matches this Python/platform')
    return min(candidates, key=lambda p: (p.name.endswith('none-any.whl'), p.name))


def call(argv, directory, environment, commands, *, json_output=True):
    started = time.monotonic()
    result = subprocess.run([str(item) for item in argv], cwd=directory, env=environment,
                            capture_output=True, text=True, check=False, timeout=180)
    commands.append({'argv': [str(item) for item in argv], 'cwd': str(directory),
                     'returncode': result.returncode, 'seconds': time.monotonic()-started,
                     'stderr': result.stderr[-6000:]})
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {argv}\n{result.stdout}\n{result.stderr}')
    return json.loads(result.stdout) if json_output else result.stdout


def run(core_wheel, native_wheel, result):
    with tempfile.TemporaryDirectory(prefix='ariadne-dual-wheel-') as temporary:
        temporary = Path(temporary).resolve()
        if temporary.is_relative_to(ROOT):
            raise RuntimeError('Wheel smoke directory must be outside the checkout')
        target, outside = temporary/'target', temporary/'outside-checkout'
        outside.mkdir()
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(target)
        environment['PYTHONNOUSERSITE'] = '1'
        environment['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
        environment['ARIADNE_PLUGIN_ROOT'] = str(ROOT)
        environment['ARIADNE_NUMERIC_BACKEND'] = 'auto'
        commands = result['commands']
        install = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
                   '--target', target, core_wheel, native_wheel]
        result['pip_stdout'] = call(install, outside, environment, commands, json_output=False)
        result['probe'] = call([sys.executable, '-c', PROBE, target, ROOT], outside, environment, commands)
        with zipfile.ZipFile(native_wheel) as archive:
            contains_extension = any(name.endswith(('.so', '.pyd')) for name in archive.namelist())
        if result['probe']['kernel']['compiled_available'] != contains_extension:
            raise RuntimeError('Loaded numeric backend does not match the installed native wheel')

        example = outside/'inputs'
        example.mkdir()
        files = ('fen4_start.json', 'fen4_alternative_start.json', 'fen4_measured.json',
                 'refine_fen4.json', 'parallel_native.json')
        for name in files:
            shutil.copyfile(ROOT/'examples/v08'/name, example/name)
        result['input_sha256'] = {name: sha256(example/name) for name in files}
        result['entry_points'] = {name: {'path': str(target/'bin'/name), 'sha256': sha256(target/'bin'/name)}
                                  for name in ('ariadne-agent', 'ariadne-xtal')}

        native_response = call([target/'bin/ariadne-xtal', 'run', example/'refine_fen4.json',
                                '--root', example], outside, environment, commands)
        if native_response.get('ok') is not True:
            raise RuntimeError(f'Native CLI reported failure: {native_response}')
        fit_file = example/native_response['result']
        fit = json.loads(fit_file.read_text())
        if not fit['success']:
            raise RuntimeError('Native CLI ran but numerical refinement did not converge')
        result['native_cli_refinement'] = {'success': fit['success'], 'metrics': fit['metrics'],
                                         'result_sha256': sha256(fit_file), 'summary': native_response['summary']}

        session = outside/'persistent-session'
        agent = target/'bin/ariadne-agent'
        call([agent, 'init', example/'parallel_native.json', session], outside, environment, commands)
        checkpoint = call([agent, 'run', session, '--steps', '2'], outside, environment, commands)
        statuses = {n['id']: n['status'] for n in checkpoint['nodes']}
        expected = {'refine_a': 'succeeded', 'refine_b': 'succeeded', 'review': 'pending', 'package': 'pending'}
        if checkpoint['actions_used'] != 2 or statuses != expected:
            raise RuntimeError(f'Unexpected checkpoint: {checkpoint}')
        final = call([agent, 'run', session], outside, environment, commands)
        if final['actions_used'] != 4 or not all(n['status'] == 'succeeded' for n in final['nodes']):
            raise RuntimeError(f'Continued workflow did not complete: {final}')
        repeated = call([agent, 'run', session], outside, environment, commands)
        if repeated['actions_used'] != final['actions_used']:
            raise RuntimeError('Completed workflow was executed again')
        with sqlite3.connect(session/'agent.sqlite') as database:
            database.row_factory = sqlite3.Row
            attempts = [dict(row) for row in database.execute('SELECT * FROM attempts ORDER BY started')]
            nodes = [dict(row) for row in database.execute('SELECT id,attempts,result FROM nodes ORDER BY rowid')]
        if any(n['attempts'] != 1 for n in nodes):
            raise RuntimeError('A completed node has more than one attempt')
        a, b = attempts[:2]
        overlap = max(a['started'], b['started']) < min(a['finished'], b['finished'])
        if not overlap:
            raise RuntimeError('The two ready native tasks did not overlap')
        outcomes = {}
        package_ok = False
        for node in nodes:
            record = json.loads(node['result'])
            if node['id'].startswith('refine_'):
                scientific = record['scientific_result']
                if not scientific['success']:
                    raise RuntimeError(f"Refinement {node['id']} did not converge")
                outcomes[node['id']] = {'success': scientific['success'], 'metrics': scientific['metrics'],
                                       'native_result_sha256': record['native_result']['sha256']}
            elif node['id'] == 'package':
                archive = session/record['package']['path']
                if sha256(archive) != record['package']['sha256']:
                    raise RuntimeError('Packaged result hash does not match its record')
                with zipfile.ZipFile(archive) as stream:
                    package_ok = stream.testzip() is None and 'manifest.json' in stream.namelist()
                if not package_ok:
                    raise RuntimeError('Result archive failed integrity check')
        result['persistent_agent'] = {'checkpoint_actions': checkpoint['actions_used'],
            'checkpoint_statuses': statuses, 'final_actions': final['actions_used'],
            'all_nodes_succeeded': True, 'ready_native_tasks_overlapped': overlap,
            'resume_reexecuted_completed_nodes': False, 'all_attempt_counts': [n['attempts'] for n in nodes],
            'wall_elapsed_seconds': final['wall_elapsed_seconds'],
            'cumulative_action_seconds': final['cumulative_action_seconds'],
            'attempts': attempts, 'outcomes': outcomes, 'package_zip_and_hash_verified': package_ok}
        result['ok'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-wheel', type=Path, default=ROOT/'dist/v0.8/ariadne-0.8.0-py3-none-any.whl')
    parser.add_argument('--native-wheel', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/v08/results/wheel_integration.json')
    args = parser.parse_args()
    core_wheel = args.core_wheel.resolve()
    native_wheel = (args.native_wheel or choose_native_wheel(ROOT/'dist/v0.8')).resolve()
    result = {'schema': 'ariadne.wheel-integration/1', 'ok': False,
              'platform': platform.platform(), 'python': sys.version,
              'scope': 'Fresh --target installation of both wheels; third-party dependencies reused from the validation environment',
              'not_tested_here': 'Clean-machine dependency bootstrap, Windows execution or full DIALS image processing',
              'wheels': [{'file': p.name, 'sha256': sha256(p), 'bytes': p.stat().st_size}
                         for p in (core_wheel, native_wheel)], 'commands': []}
    try:
        run(core_wheel, native_wheel, result)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'ok': result['ok'], 'output': str(args.output),
                      'all_ariadne_modules_from_target': result['probe']['all_ariadne_modules_from_target'],
                      'numeric_backend': result['probe']['kernel']['resolved'],
                      'concurrent_tasks': result['persistent_agent']['ready_native_tasks_overlapped'],
                      'actions_before_after_resume': [result['persistent_agent']['checkpoint_actions'],
                                                      result['persistent_agent']['final_actions']]}, indent=2))


if __name__ == '__main__':
    main()
