# Launchpad — Enterprise Agent Platform

**Status: all three phases shipped, plus config-file + CLI launch and a P0 hardening pass.**
23 tests, no network required. See `README.md` to run it.

Verified at 400 concurrent invokes across 4 uvicorn workers with memory enabled:
0 failures, 0 tracebacks, 0 lock errors, clean shutdown.

### Fixed in the hardening pass

| Was | Now |
|---|---|
| `InMemorySaver` -- conversation history reset whenever a turn hit another worker | SQLite-backed `AsyncSqliteSaver` shared by every worker; thread ids namespaced per agent |
| Blocking `sqlite3` calls inside `async` handlers stalled the event loop | every store call offloaded via `asyncio.to_thread` |
| One `httpx.AsyncClient` per agent -- leaked its pool on cache eviction | one shared client, closed by the app lifespan |
| `http_get` buffered the whole response before truncating | streamed with a 200KB cap |
| Upstream `Retry-After` slept on verbatim | clamped to 10s, non-numeric tolerated |
| *(found while load testing)* `PRAGMA journal_mode` ignores `busy_timeout`, so 2 of 4 workers died at boot | tolerated; WAL is persistent, whichever worker wins sets it for all |
| *(found while load testing)* one `sqlite3` connection shared across threadpool threads -> `InterfaceError` | one connection per thread |
| *(found while load testing)* default `isolation_level` pinned each worker to a stale read snapshot | `isolation_level=None` (autocommit) |
| *(found while load testing)* `_agent()` read the config twice; the second read raced to `None` -> 500 | read once, pass it to the builder |

**One line:** users POST a config, platform returns a live agent endpoint. No agent code written by the user.

```
POST /agents        { name, runtime, model, tools, prompt, limits }  -> agent_id
POST /agents/{id}/invoke   { input }        -> output      (SSE for stream)
GET  /agents/{id}/runs/{run_id}             -> trace, tokens, latency
```

Supported runtimes: `langchain_react`, `databricks_mlflow`.

## The one architectural decision

Every runtime builds a **LangChain `Runnable`**. That's the common interface —
`.invoke()`, `.stream()`, `.batch()`, async, and callbacks all come free. A Databricks
serving endpoint wraps in `ChatDatabricks` (or 15 lines of `RunnableLambda` + httpx),
so it plugs into the same executor as the ReAct agent.

No adapter layer, no `BaseAgent` ABC, no factory classes. The registry is a dict:

```python
RUNTIMES: dict[str, Callable[[AgentConfig], Runnable]] = {}


def runtime(name):  # decorator
    def deco(fn):
        RUNTIMES[name] = fn
        return fn

    return deco


@lru_cache(maxsize=256)  # build once, reuse across requests
def get_agent(cfg_hash: str) -> Runnable:
    cfg = store.load(cfg_hash)
    return RUNTIMES[cfg.runtime](cfg)
```

Adding a runtime = one decorated function. That's the platform.

## Stack

| Need | Choice | Why not the other thing |
|---|---|---|
| API | FastAPI | Pydantic validation is the config schema. Free. |
| Config validation | Pydantic v2 discriminated union | Per-runtime fields validated at POST, not at invoke |
| Store | SQLite (`sqlite3`, stdlib) | Postgres when >1 replica. Agents are content-addressed by config hash. |
| Auth | API key header + `hmac.compare_digest` | OIDC when there are real tenants |
| Observability | `runs` table (input, output, error, latency) | MLflow autolog is one line but drags in ~200MB; add when someone needs step-level spans |
| Packaging | `uv` + `pyproject.toml`, ruff, pytest | — |

---

## Phase 1 — Platform skeleton (no agents yet) ✅

**Goal:** config in, endpoint out, proven with a trivial `echo` runtime.

```
launchpad/
  config.py      # AgentConfig pydantic models, discriminated on `runtime`
  registry.py    # RUNTIMES dict + @runtime decorator + get_agent()
  store.py       # sqlite3: save/load/list configs, run records
  api.py         # FastAPI routes: create, list, invoke, stream, runs
  runtimes/
    echo.py      # @runtime("echo") -> RunnableLambda(lambda x: x)
  auth.py        # X-API-Key dependency
tests/test_platform.py
```

Deliverables:
- Config schema with per-runtime discriminated union; bad config → 422 with field path.
- `POST /agents` persists, `POST /agents/{id}/invoke` returns output + `run_id`.
- SSE streaming endpoint (`/invoke?stream=true`) using `Runnable.astream`.
- Every run row: `run_id, agent_id, input, output, latency_ms, tokens, error`.
- One pytest that creates an echo agent and invokes it. TestClient, no fixtures.

Guardrails that are NOT skipped (trust boundary): API-key auth, config validation,
per-agent `timeout_s` + `max_iterations`, secrets read from env/`.env` and never
persisted in the config row.

**Done when:** `curl` creates and invokes an agent end-to-end with zero LangChain imported.

---

## Phase 2 — LangChain ReAct runtime ✅

`runtimes/langchain_react.py`, one function:

```python
@runtime("langchain_react")
def build(cfg: LangChainConfig) -> Runnable:
    model = init_chat_model(cfg.model.name, temperature=cfg.model.temperature)
    tools = [TOOLS[t] for t in cfg.tools]  # allowlist, KeyError -> 400 at create
    return create_react_agent(
        model, tools, prompt=cfg.system_prompt, checkpointer=InMemorySaver() if cfg.memory else None
    )
```

Use LangGraph's prebuilt `create_react_agent` — do not hand-roll the ReAct loop.

Work:
- Tool registry: same decorator pattern. Shipped `http_get` (with an SSRF guard) and `utc_now`; the catalog grows by adding a decorated function. Tools are **allowlisted by name**, never eval'd from config — that's the injection boundary.
- `thread_id` passthrough for multi-turn memory.
- Recursion/timeout limits mapped from `cfg.limits`.
- Tests: end-to-end invoke against a fake chat model (no key, no network); unknown tool name rejected at create time with the allowlist in the error.

**Done when:** a YAML config with 2 tools produces a working multi-turn agent, traced.

---

## Phase 3 — Databricks MLflow runtime ✅

`runtimes/databricks_mlflow.py`:

Shipped as ~90 lines of `httpx` against `/serving-endpoints/{name}/invocations`
rather than `databricks-langchain` + `mlflow` (~200MB) for one HTTP POST. Both
endpoint shapes go through the same code path, and the whole thing is testable
offline. Swap in `ChatDatabricks` if the workspace's auth story outgrows a PAT.

Work:
- Auth: `DATABRICKS_HOST` + `DATABRICKS_TOKEN` from env, never in the config row.
- `mode: "chat"` normalises ChatAgent (`messages`) and OpenAI (`choices`) responses;
  `mode: "raw"` handles MLflow pyfunc (`inputs` / `predictions`).
- Endpoint errors → 502; 429/503 retried with backoff honouring `Retry-After`,
  then surfaced as a 429 with the header intact.
- Test: httpx `MockTransport` returns canned endpoint responses; asserts the **same
  invoke contract as Phase 2**. That test is the point — it proves the Runnable
  abstraction holds across both runtimes instead of just claiming it does.

**Done when:** the same `POST /agents/{id}/invoke` call works against a Databricks
endpoint and a LangChain agent, with identical request/response shape.

---

## Deliberately skipped

| Skipped | Add when |
|---|---|
| MLflow tracing / step-level spans | run-level latency + errors stop being enough |
| Kubernetes, autoscaling, worker queue | invoke p95 > request timeout, or >1 box |
| Multi-tenancy / RBAC / orgs | second paying tenant |
| Vector store / RAG built in | a config actually asks for a retriever tool |
| Web UI | the API is being used and someone asks |
| Config hot-reload / `apply` diffing + deletes | configs change often enough that restart-to-reload hurts |
| Postgres, Redis, Celery | SQLite lock contention shows up in traces |
| Plugin entry-points system | a 4th runtime, added by someone outside the repo |

## Resume bullets this produces

- Built a config-driven agent platform (FastAPI/Pydantic) that provisions and serves
  production agents from a declarative spec — no per-agent code.
- Designed a pluggable runtime registry over LangChain's `Runnable` interface,
  unifying LangGraph ReAct agents and Databricks MLflow serving endpoints behind one
  invoke/stream API; new runtime = one function.
- Added per-run tracing, token accounting, tool allowlisting, and timeout/recursion
  limits via MLflow autologging.
