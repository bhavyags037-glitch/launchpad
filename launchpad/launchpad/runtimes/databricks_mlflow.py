"""Databricks Model Serving endpoints as platform agents.

ponytail: talks to `/serving-endpoints/{name}/invocations` over httpx rather than
pulling in `databricks-langchain` + `mlflow` (~200MB) for one HTTP POST. Swap in
`ChatDatabricks` if the workspace's auth story outgrows a PAT.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException

from launchpad.config import DatabricksConfig
from launchpad.registry import runtime

RETRY_STATUS = {429, 502, 503, 504}
MAX_RETRY_SLEEP = 10.0  # an upstream Retry-After must not pin a worker

# ponytail: one client for every Databricks agent rather than one each. A per-agent
# client leaks its connection pool when the agent cache evicts it; a shared pool
# cannot. Closed by the app's lifespan.
_http: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(limits=httpx.Limits(max_connections=100))
    return _http


async def close_client() -> None:
    global _http
    if _http is not None and not _http.is_closed:
        await _http.aclose()
    _http = None


def _retry_after(response: httpx.Response, attempt: int) -> float:
    """Upstream-controlled, so clamp it. A bad header must not stall the worker."""
    try:
        delay = float(response.headers.get("retry-after", 2**attempt))
    except ValueError:
        delay = float(2**attempt)
    return max(0.0, min(delay, MAX_RETRY_SLEEP))


def _messages(x: Any) -> list[dict]:
    if isinstance(x, str):
        return [{"role": "user", "content": x}]
    if isinstance(x, dict) and "messages" in x:
        return x["messages"]
    if isinstance(x, list):
        return x
    raise ValueError("input must be a string, a message list, or {'messages': [...]}")


def _extract(body: Any) -> Any:
    """Normalise the three shapes Databricks endpoints return into one `output`."""
    if isinstance(body, dict):
        if choices := body.get("choices"):
            return choices[0].get("message", {}).get("content")
        if messages := body.get("messages"):  # MLflow ChatAgent
            return messages[-1].get("content")
        if "predictions" in body:
            return body["predictions"]
    return body


class _ServingEndpoint:
    def __init__(self, cfg: DatabricksConfig) -> None:
        host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        token = os.environ.get("DATABRICKS_TOKEN", "")
        if not host or not token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set")
        self.cfg = cfg
        self.url = f"{host}/serving-endpoints/{cfg.endpoint}/invocations"
        self.headers = {"Authorization": f"Bearer {token}"}

    def _payload(self, input_: Any) -> dict:
        if self.cfg.mode == "raw":
            return input_ if isinstance(input_, dict) else {"inputs": input_}
        return {"messages": _messages(input_)}

    async def ainvoke(self, input_: Any, config: dict | None = None) -> Any:
        payload = self._payload(input_)
        for attempt in range(self.cfg.retries + 1):
            r = await client().post(
                self.url, json=payload, headers=self.headers, timeout=self.cfg.limits.timeout_s
            )
            if r.status_code not in RETRY_STATUS:
                break
            if attempt == self.cfg.retries:
                raise HTTPException(
                    429 if r.status_code == 429 else 502,
                    f"endpoint {self.cfg.endpoint} unavailable ({r.status_code})",
                    headers={"Retry-After": str(int(_retry_after(r, attempt)))},
                )
            await asyncio.sleep(_retry_after(r, attempt))

        if r.is_error:
            raise HTTPException(
                502, f"endpoint {self.cfg.endpoint} returned {r.status_code}: {r.text[:200]}"
            )
        return {"output": _extract(r.json())}

    async def astream(self, input_: Any, config: dict | None = None) -> AsyncIterator[Any]:
        # ponytail: one chunk. Set "stream": true in the payload and parse SSE if
        # token-level streaming from the endpoint is ever actually needed.
        yield await self.ainvoke(input_, config)


@runtime("databricks_mlflow")
def build(cfg: DatabricksConfig) -> _ServingEndpoint:
    return _ServingEndpoint(cfg)
