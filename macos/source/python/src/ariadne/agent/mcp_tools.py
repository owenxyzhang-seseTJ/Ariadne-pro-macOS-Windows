"""Optional MCP bridge. CLI/numerical kernels do not require an MCP runtime."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from . import async_runs, engine
from .store import Store, scientific_summary
from .util import dumps, pointer


class Reply(BaseModel):
    data: dict[str, Any] = Field(description='Structured agent state or operation result')


def register(mcp):
    @mcp.tool()
    def agent_capabilities() -> Reply:
        return Reply(data=engine.capabilities())

    @mcp.tool()
    def agent_init(request: dict[str, Any], session: str) -> Reply:
        return Reply(data=engine.initialize(request, session))

    @mcp.tool()
    def agent_step(session: str) -> Reply:
        """Run at most one bounded scientific action; call again after inspecting state."""
        return Reply(data=engine.run(session, max_steps=1))

    @mcp.tool()
    def agent_status(session: str) -> Reply:
        return Reply(data=engine.status(session))

    @mcp.tool()
    def agent_cancel(session: str) -> Reply:
        return Reply(data=engine.cancel(session))

    @mcp.tool()
    def agent_resume(session: str) -> Reply:
        return Reply(data=engine.resume(session))

    @mcp.tool()
    def agent_retry(session: str, node: str) -> Reply:
        return Reply(data=engine.retry(session, node))

    @mcp.tool()
    def agent_export(session: str, destination: str, include_inputs: bool = False) -> Reply:
        return Reply(data=engine.export_session(session, destination, include_inputs=include_inputs))

    @mcp.tool()
    def agent_submit(session: str, request_key: str, max_steps: int | None = None) -> Reply:
        """Start a detached bounded run. Reusing the same request_key returns its run id."""
        return Reply(data=async_runs.submit(session, request_key, max_steps))

    @mcp.tool(annotations={'readOnlyHint': True})
    def agent_run_status(session: str, run_id: str) -> Reply:
        """Read a detached run by its stable identifier."""
        return Reply(data=async_runs.status(session, run_id))

    @mcp.tool(annotations={'readOnlyHint': True})
    def agent_events(session: str, after: int = 0, limit: int = 100) -> Reply:
        """Read a bounded page of scientific state events from a session cursor."""
        return Reply(data=async_runs.events(session, after, limit))

    @mcp.tool(annotations={'readOnlyHint': True})
    def agent_result(session: str, node: str, json_pointer: str | None = None, max_chars: int = 12000) -> Reply:
        """Read one persisted node's scientific summary or a bounded selected result."""
        if not 1 <= max_chars <= 20000:
            raise ValueError('max_chars must be in 1..20000')
        with closing(Store(Path(session))) as store:
            item = store.node(node)
        result = item['result']
        details = {'node': node, 'state': item['status'], 'error': item['error'],
                   'summary': scientific_summary(result)}
        if result is not None and json_pointer is not None:
            value = pointer(result, json_pointer)
            serialized = dumps(value)
            if len(serialized) <= max_chars:
                details['result'] = value
            else:
                details.update(preview=serialized[:max_chars], truncated=True)
        return Reply(data=details)

    @mcp.tool()
    def agent_cancel_run(session: str, run_id: str) -> Reply:
        """Request cancellation only while the identified detached run is active."""
        return Reply(data=async_runs.cancel(session, run_id))
