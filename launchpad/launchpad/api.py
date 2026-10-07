"""HTTP surface: create an agent from a config, invoke it, read its runs."""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from launchpad import checkpoints, configs, registry, store
from launchpad.auth import require_key
from launchpad.config import AgentConfig
from launchpad.registry import Runnable, UnknownRuntime, get_agent
from launchpad.runtimes import databricks_mlflow

# uvicorn owns the handlers; borrowing its logger is how startup lines get seen.
log = logging.getLogger("uvicorn.error")


def _create(cfg: AgentConfig) -> str:
    """Build the agent (so a broken config fails now, not at 3am) then persist it."""
    registry.build(cfg)
    return store.save_agent(cfg)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Open shared resources, then launch every agent declared in $LAUNCHPAD_CONFIG."""
    await checkpoints.start()  # before seeding: an agent with memory needs it at build time
    if path := os.environ.get("LAUNCHPAD_CONFIG"):
        for cfg in configs.load(path):
            log.info("launched %s [%s] -> %s", cfg.name, cfg.runtime, _create(cfg))
    try:
        yield
    finally:
        await databricks_mlflow.close_client()
        await checkpoints.close()


app = FastAPI(
    title="Launchpad",
    description="Config-driven agent platform",
    version="0.1.0",
    lifespan=lifespan,
)


class InvokeRequest(BaseModel):
    input: Any
    stream: bool = False
    thread_id: str | None = None  # multi-turn memory, for runtimes that checkpoint


def _run_config(agent_id: str, req: "InvokeRequest") -> dict | None:
    # Namespaced by agent: every agent shares one checkpoint store, so a caller
    # reusing "session-1" across two agents must not read the other's history.
    if not req.thread_id:
        return None
    return {"configurable": {"thread_id": f"{agent_id}:{req.thread_id}"}}


def _agent(agent_id: str) -> tuple[AgentConfig, Runnable]:
    cfg = store.load_agent(agent_id)
    if cfg is None:
        raise HTTPException(404, f"no agent {agent_id}")
    return cfg, get_agent(agent_id, cfg)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "runtimes": sorted(registry.RUNTIMES)}


@app.post("/agents", status_code=201, dependencies=[Depends(require_key)])
def create_agent(cfg: AgentConfig) -> dict:
    try:
        agent_id = _create(cfg)
    except UnknownRuntime as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(400, f"cannot build agent: {e}") from e
    return {"agent_id": agent_id, "name": cfg.name, "runtime": cfg.runtime}


@app.get("/agents", dependencies=[Depends(require_key)])
def list_agents() -> list[dict]:
    return store.list_agents()


@app.get("/agents/{agent_id}", dependencies=[Depends(require_key)])
def get_agent_config(agent_id: str) -> dict:
    cfg, _ = _agent(agent_id)
    return {"agent_id": agent_id, "config": cfg.model_dump()}


@app.post("/agents/{agent_id}/invoke", dependencies=[Depends(require_key)])
async def invoke(agent_id: str, req: InvokeRequest) -> Any:
    cfg, agent = await asyncio.to_thread(_agent, agent_id)
    run_config = _run_config(agent_id, req)
    if req.stream:
        return StreamingResponse(
            _stream(agent_id, cfg, agent, req.input, run_config),
            media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    try:
        output = await asyncio.wait_for(agent.ainvoke(req.input, run_config), cfg.limits.timeout_s)
    except TimeoutError as e:
        await _record(agent_id, req.input, error="timeout", latency_ms=_ms(t0))
        raise HTTPException(504, f"agent exceeded timeout_s={cfg.limits.timeout_s}") from e
    except HTTPException as e:  # runtimes may map upstream failures themselves
        await _record(agent_id, req.input, error=e.detail, latency_ms=_ms(t0))
        raise
    except Exception as e:
        await _record(agent_id, req.input, error=repr(e), latency_ms=_ms(t0))
        raise HTTPException(502, f"agent failed: {e}") from e

    run_id = await _record(agent_id, req.input, output=output, latency_ms=_ms(t0))
    return {"run_id": run_id, "output": output}


@app.get("/runs/{run_id}", dependencies=[Depends(require_key)])
def get_run(run_id: str) -> dict:
    run = store.load_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run


async def _record(*args: Any, **kwargs: Any) -> str:
    """sqlite3 blocks; inside an async handler that stalls every other request."""
    return await asyncio.to_thread(store.record_run, *args, **kwargs)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def _sse(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, default=str)}\n\n"


async def _stream(
    agent_id: str, cfg: AgentConfig, agent: Runnable, payload: Any, run_config: dict | None
) -> AsyncIterator[str]:
    t0, chunks = time.perf_counter(), []
    try:
        async with asyncio.timeout(cfg.limits.timeout_s):
            async for chunk in agent.astream(payload, run_config):
                chunks.append(chunk)
                yield _sse(chunk)
    except Exception as e:  # noqa: BLE001 - a half-sent SSE stream must end as an event, not a traceback
        err = "timeout" if isinstance(e, TimeoutError) else repr(e)
        await _record(agent_id, payload, error=err, latency_ms=_ms(t0))
        yield _sse({"error": err}, event="error")
        return
    run_id = await _record(agent_id, payload, output=chunks, latency_ms=_ms(t0))
    yield _sse({"run_id": run_id}, event="done")
