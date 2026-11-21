"""Measure local MCP responses during a batch of CIF audits."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import numpy as np

from ariadne.agent import engine


async def measure(root, fixture):
    nodes = [engine.node(f'audit_{i}', 'crystal.audit', {'cif': engine.input_ref('cif')})
             for i in range(100)]
    engine.initialize({'mode': 'custom', 'experts': [], 'inputs': {'cif': str(fixture)}, 'nodes': nodes,
                       'scheduler': {'max_workers': 4, 'cpu_threads': 4}}, root)
    program = StdioServerParameters(command=sys.executable, args=['-m', 'ariadne.server'], env=dict(os.environ))
    async with stdio_client(program) as (read, write), ClientSession(read, write) as client:
        await client.initialize()

        async def request(tool, arguments):
            start = time.perf_counter()
            response = await client.call_tool(tool, arguments)
            milliseconds = (time.perf_counter() - start) * 1000
            if response.isError:
                raise RuntimeError(str(response.content))
            return response.structuredContent['data'], milliseconds

        arguments = {'session': str(root), 'request_key': 'batch-100'}
        submitted, first = await request('agent_submit', arguments)
        run_id = submitted['id']
        retries = []
        for _ in range(20):
            result, elapsed = await request('agent_submit', arguments)
            if result['id'] != run_id:
                raise RuntimeError('Repeated submissions returned different run IDs')
            retries.append(elapsed)
        reads = []
        deadline = time.monotonic() + 95
        while time.monotonic() < deadline:
            result, elapsed = await request('agent_run_status', {'session': str(root), 'run_id': run_id})
            reads.append(elapsed)
            if result['status'] not in {'queued', 'running'}:
                break
            await asyncio.sleep(.05)
        snapshot = engine.status(root)
        if result['status'] != 'completed' or any(n['status'] != 'succeeded' for n in snapshot['nodes']):
            raise RuntimeError('The benchmark run did not finish all audit nodes')
        return {'nodes': len(nodes), 'max_workers': 4, 'first_submit_ms': first,
                'p95_idempotent_submit_ms': float(np.percentile(retries, 95)),
                'status_samples': len(reads), 'p95_run_status_ms': float(np.percentile(reads, 95)),
                'runner_wall_seconds': snapshot['wall_elapsed_seconds'], 'actions_used': snapshot['actions_used']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cif', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='ariadne-mcp-bench.') as folder:
        summary = asyncio.run(measure(Path(folder) / 'session', args.cif.resolve()))
    args.output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
