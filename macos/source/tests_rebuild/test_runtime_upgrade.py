"""Reconnect, cache reuse, and process ownership across the OS boundary."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import os
import socket
import subprocess
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
import httpx
import pytest

from ariadne.agent import async_runs, engine
from ariadne.agent.process import spawn_owned, stop_process_group
from ariadne.agent.store import Store
from ariadne.agent.util import exclusive_lock
from ariadne.server import pid_alive


def wait_for(condition, *, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        time.sleep(.03)
    raise AssertionError('condition did not become true within the test budget')


def test_owned_process_terminates_a_live_descendant(tmp_path):
    record = tmp_path / 'child.pid'
    argv = [sys.executable, '-c',
            'import subprocess,sys,time; from pathlib import Path; '
            'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
            'Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)', str(record)]
    process = spawn_owned(argv, cwd=tmp_path)
    try:
        def read_pid():
            return int(record.read_text()) if record.exists() and record.stat().st_size else None
        pid = wait_for(read_pid)
        assert pid_alive(pid)
    finally:
        stop_process_group(process)
    assert process.poll() is not None
    wait_for(lambda: not pid_alive(pid))


def test_runner_lock_contends_across_processes(tmp_path):
    lock = tmp_path / 'runner.lock'
    argv = [sys.executable, '-c',
            'from ariadne.agent.util import exclusive_lock; from pathlib import Path; import sys; '
            'guard = exclusive_lock(Path(sys.argv[1])); guard.__enter__(); guard.__exit__(None,None,None)',
            str(lock)]
    with exclusive_lock(lock):
        other = subprocess.run(argv, text=True, capture_output=True, timeout=10)
        assert other.returncode != 0 and 'another runner holds this session' in other.stderr
    subprocess.run(argv, check=True, timeout=10)


@pytest.mark.parametrize('transport', ['stdio', 'streamable-http'])
def test_mcp_run_survives_disconnect_and_reuses_only_identical_science(tmp_path, transport):
    if os.name == 'nt' and transport == 'stdio':
        pytest.skip('Windows stdio clients own the server process tree; detach through loopback HTTP')
    fixture = Path(__file__).resolve().parents[1] / 'native/validation/public_data/2242624.cif'
    request = {'mode': 'audit', 'inputs': {'cif': str(fixture)}, 'experts': [],
               'cache': {'directory': str(tmp_path / 'shared-cache'), 'max_mib': 1}}
    roots = [tmp_path / name for name in ('first project', 'second project', 'changed project')]
    engine.initialize(request, roots[0])
    settings = StdioServerParameters(command=sys.executable, args=['-m', 'ariadne.server'], env=dict(os.environ))
    host = '127.0.0.1'
    server_process = None
    if transport == 'streamable-http':
        with socket.socket() as listener:
            listener.bind((host, 0))
            port = listener.getsockname()[1]
        server_process = subprocess.Popen(
            [sys.executable, '-m', 'ariadne.server', '--transport', transport, '--port', str(port)],
            stdin=subprocess.DEVNULL)

    @asynccontextmanager
    async def connect():
        async with httpx.AsyncClient(trust_env=False) as client_http:
            connection = (stdio_client(settings) if transport == 'stdio'
                          else streamable_http_client(f'http://{host}:{port}/mcp', http_client=client_http))
            async with connection as streams:
                async with ClientSession(streams[0], streams[1]) as client:
                    await client.initialize()
                    yield client

    async def call(client, name, values):
        output = await client.call_tool(name, values)
        assert not output.isError, output.content
        return output.structuredContent['data']

    async def start():
        async with connect() as client:
            tools = await client.list_tools()
            assert {'agent_submit', 'agent_events', 'agent_cancel_run'} <= {x.name for x in tools.tools}
            values = {'session': str(roots[0]), 'request_key': 'run-once', 'max_steps': 1}
            first, retry = await asyncio.gather(call(client, 'agent_submit', values),
                                                call(client, 'agent_submit', values))
            assert first['id'] == retry['id']
            return first['id']

    async def reconnect(run_id):
        async with connect() as client:
            found = await call(client, 'agent_run_status', {'session': str(roots[0]), 'run_id': run_id})
            assert found['status'] == 'completed'
            limits = {'session': str(roots[0]), 'limit': 2}
            first = await call(client, 'agent_events', limits)
            second = await call(client, 'agent_events', {**limits, 'after': first['next_cursor']})
            assert first['more'] and first['next_cursor'] < second['events'][0]['seq']

    def ready():
        with socket.socket() as probe:
            return probe.connect_ex((host, port)) == 0

    try:
        if server_process:
            wait_for(ready)
        run_id = asyncio.run(start())
        with pytest.raises(ValueError, match='different max_steps'):
            async_runs.submit(roots[0], 'run-once', 2)
        first_run = wait_for(lambda: terminal(roots[0], run_id))
        assert first_run['status'] == 'completed', first_run
        assert first_run['id'] == async_runs.submit(roots[0], 'run-once', 1)['id']
        assert engine.status(roots[0])['actions_used'] == 1
        assert audit_cache_hit(roots[0]) is False
        asyncio.run(reconnect(run_id))
    finally:
        if server_process:
            server_process.terminate()
            server_process.wait(timeout=10)
    engine.initialize(request, roots[1])
    engine.run(roots[1], max_steps=1)
    assert audit_cache_hit(roots[1]) is True
    changed = {**request, 'settings': {'expected_elements': ['Fe']}}
    engine.initialize(changed, roots[2])
    engine.run(roots[2], max_steps=1)
    assert audit_cache_hit(roots[2]) is False


def terminal(root, run_id):
    status = async_runs.status(root, run_id)
    return status if status['status'] not in async_runs.ACTIVE else None


def test_attempt_reservation_recovers_without_overwriting_orphaned_outputs(tmp_path):
    root = tmp_path / 'session'
    fixture = Path(__file__).resolve().parents[1] / 'native/validation/public_data/2242624.cif'
    engine.initialize({'mode': 'audit', 'inputs': {'cif': str(fixture)}, 'experts': []}, root)
    orphan = root / 'attempts/audit/001'
    orphan.mkdir(parents=True)
    file = orphan / 'before-crash.log'
    file.write_text('retained', encoding='utf-8')
    result = engine.run(root, max_steps=1)
    assert result['nodes'][0]['status'] == 'succeeded'
    assert file.read_text(encoding='utf-8') == 'retained'
    assert len(list(orphan.parent.glob('001-*'))) == 1


def audit_cache_hit(root):
    store = Store(root)
    try:
        assert store.verify_events()
        node = store.node('audit')
        assert node['status'] == 'succeeded', node
        return node['result']['stage_cache']['hit']
    finally:
        store.close()
