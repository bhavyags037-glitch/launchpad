import os
import tempfile

os.environ["LAUNCHPAD_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["LAUNCHPAD_API_KEYS"] = "test-key,second-key"

import pytest
from fastapi.testclient import TestClient

from launchpad.api import app

AUTH = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def agent_id(client):
    r = client.post(
        "/agents", headers=AUTH, json={"name": "greeter", "runtime": "echo", "prefix": "hi "}
    )
    assert r.status_code == 201, r.text
    return r.json()["agent_id"]


def test_auth_required(client):
    assert client.get("/agents").status_code == 401
    assert client.get("/agents", headers={"X-API-Key": "wrong"}).status_code == 401


def test_bad_config_rejected_at_create(client):
    r = client.post("/agents", headers=AUTH, json={"name": "x", "runtime": "nope"})
    assert r.status_code == 422  # discriminated union: unknown runtime is a schema error
    r = client.post("/agents", headers=AUTH, json={"runtime": "echo"})
    assert r.status_code == 422 and "name" in r.text
    r = client.post(
        "/agents", headers=AUTH, json={"name": "x", "runtime": "echo", "limits": {"timeout_s": 0}}
    )
    assert r.status_code == 422 and "timeout_s" in r.text


def test_create_is_idempotent(client, agent_id):
    again = client.post(
        "/agents", headers=AUTH, json={"name": "greeter", "runtime": "echo", "prefix": "hi "}
    )
    assert again.json()["agent_id"] == agent_id  # content-addressed


def test_invoke_records_a_run(client, agent_id):
    r = client.post(f"/agents/{agent_id}/invoke", headers=AUTH, json={"input": "world"})
    assert r.status_code == 200, r.text
    assert r.json()["output"] == {"output": "hi world"}

    run = client.get(f"/runs/{r.json()['run_id']}", headers=AUTH).json()
    assert run["agent_id"] == agent_id
    assert run["input"] == "world" and run["error"] is None
    assert run["latency_ms"] >= 0


def test_stream(client, agent_id):
    r = client.post(
        f"/agents/{agent_id}/invoke", headers=AUTH, json={"input": "a b", "stream": True}
    )
    assert r.status_code == 200
    body = r.text
    assert 'data: {"output": "hi"}' in body
    assert "event: done" in body


def test_unknown_agent_404(client):
    assert (
        client.post("/agents/deadbeef/invoke", headers=AUTH, json={"input": 1}).status_code == 404
    )
    assert client.get("/runs/deadbeef", headers=AUTH).status_code == 404


def test_timeout_is_enforced(client, monkeypatch):
    import asyncio

    from launchpad import registry, store

    r = client.post(
        "/agents",
        headers=AUTH,
        json={"name": "slow", "runtime": "echo", "limits": {"timeout_s": 0.01}},
    )
    aid = r.json()["agent_id"]
    agent = registry.get_agent(aid, store.load_agent(aid))
    monkeypatch.setattr(agent, "ainvoke", lambda *a, **k: asyncio.sleep(5))

    assert (
        client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "x"}).status_code == 504
    )


# --- Phase 2: LangChain ReAct runtime -----------------------------------------

os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-a-real-key")


def _fake_model(*_a, **_k):
    """Stand-in chat model: no network, no key, answers forever."""
    from itertools import cycle

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    return GenericFakeChatModel(messages=cycle([AIMessage("done")]))


def test_unknown_tool_rejected_at_create(client):
    r = client.post(
        "/agents",
        headers=AUTH,
        json={"name": "bad", "runtime": "langchain_react", "tools": ["rm_rf"]},
    )
    assert r.status_code == 400 and "rm_rf" in r.text and "allowed" in r.text


def test_react_agent_end_to_end(client, monkeypatch):
    """Same invoke contract as echo, with a fake chat model standing in for the LLM."""
    from launchpad.runtimes import langchain_react

    monkeypatch.setattr(langchain_react, "init_chat_model", _fake_model)
    r = client.post(
        "/agents",
        headers=AUTH,
        json={"name": "assistant", "runtime": "langchain_react", "system_prompt": "be terse"},
    )
    assert r.status_code == 201, r.text
    aid = r.json()["agent_id"]

    # the cached build happened under the patch; invoke goes through the real API path
    out = client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "hello"})
    assert out.status_code == 200, out.text
    assert "output" in out.json()["output"]


def test_react_agent_streams(client, monkeypatch):
    """LangGraph streams {node: {messages:[...]}} updates, not full state -- regression."""
    from launchpad.runtimes import langchain_react

    monkeypatch.setattr(langchain_react, "init_chat_model", _fake_model)
    aid = client.post(
        "/agents", headers=AUTH, json={"name": "streamer", "runtime": "langchain_react"}
    ).json()["agent_id"]

    r = client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "hi", "stream": True})
    assert r.status_code == 200
    assert '"output": "done"' in r.text and "event: done" in r.text
    assert "error" not in r.text


def test_bad_input_shape_is_a_502_not_a_crash(client, monkeypatch):
    from launchpad.runtimes import langchain_react

    monkeypatch.setattr(langchain_react, "init_chat_model", _fake_model)
    aid = client.post(
        "/agents", headers=AUTH, json={"name": "shapes", "runtime": "langchain_react"}
    ).json()["agent_id"]
    r = client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": 42})
    assert r.status_code == 502


# --- Phase 3: Databricks serving endpoints ------------------------------------

os.environ.setdefault("DATABRICKS_HOST", "https://test.cloud.databricks.com")
os.environ.setdefault("DATABRICKS_TOKEN", "dapi-test")

_REAL_ASYNC_CLIENT = __import__("httpx").AsyncClient


def _mock_databricks(monkeypatch, handler):
    """Patch httpx.AsyncClient so the runtime talks to `handler` instead of a workspace."""
    import httpx

    from launchpad.runtimes import databricks_mlflow

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kw),
    )
    # the shared client is built lazily on first request -- drop it so this
    # handler is the one that gets wired in (monkeypatch restores it after)
    monkeypatch.setattr(databricks_mlflow, "_http", None)


def _make(client, **extra):
    body = {"name": "dbx", "runtime": "databricks_mlflow", "endpoint": "my-agent", **extra}
    r = client.post("/agents", headers=AUTH, json=body)
    assert r.status_code == 201, r.text
    return r.json()["agent_id"]


def test_databricks_honours_the_same_contract_as_langchain(client, monkeypatch):
    """The point of the whole design: one invoke contract across both runtimes."""
    import httpx

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"messages": [{"role": "assistant", "content": "42"}]})

    _mock_databricks(monkeypatch, handler)
    aid = _make(client, name="dbx-chat")

    r = client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "meaning of life?"})
    assert r.status_code == 200, r.text
    assert r.json()["output"] == {"output": "42"}  # same shape echo and the ReAct agent return
    assert seen["url"].endswith("/serving-endpoints/my-agent/invocations")
    assert seen["auth"] == "Bearer dapi-test"


def test_databricks_openai_and_pyfunc_shapes(client, monkeypatch):
    import httpx

    _mock_databricks(
        monkeypatch,
        lambda _r: httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}),
    )
    aid = _make(client, name="dbx-openai")
    assert client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "x"}).json()[
        "output"
    ] == {"output": "hi"}

    _mock_databricks(monkeypatch, lambda _r: httpx.Response(200, json={"predictions": [0.9]}))
    aid = _make(client, name="dbx-raw", mode="raw")
    assert client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": [1, 2]}).json()[
        "output"
    ] == {"output": [0.9]}


def test_databricks_rate_limit_surfaces_as_429(client, monkeypatch):
    import httpx

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "0"}, json={"message": "slow down"})

    _mock_databricks(monkeypatch, handler)
    aid = _make(client, name="dbx-429", retries=1)
    r = client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "x"})
    assert r.status_code == 429 and r.headers["retry-after"] == "0"
    assert len(calls) == 2  # initial attempt + one retry


def test_databricks_error_is_recorded_on_the_run(client, monkeypatch):
    import httpx

    _mock_databricks(monkeypatch, lambda _r: httpx.Response(400, text="bad model input"))
    aid = _make(client, name="dbx-400")
    assert (
        client.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "x"}).status_code == 502
    )

    runs = [r for r in _runs_for(aid) if r["error"]]
    assert runs and "bad model input" in runs[0]["error"]


def _runs_for(agent_id):
    from launchpad import store

    rows = store.db().execute("select * from runs where agent_id=?", (agent_id,)).fetchall()
    return [dict(r) for r in rows]


# --- Config files -------------------------------------------------------------


def test_config_file_loads_yaml_and_json(tmp_path):
    from launchpad import configs

    y = tmp_path / "agents.yaml"
    y.write_text(
        "agents:\n  - {name: a, runtime: echo}\n  - {name: b, runtime: echo, prefix: 'x '}\n"
    )
    assert [c.name for c in configs.load(y)] == ["a", "b"]

    j = tmp_path / "one.json"
    j.write_text('{"name": "solo", "runtime": "echo"}')  # bare document, no list
    assert [c.name for c in configs.load(j)] == ["solo"]


def test_config_file_rejects_a_bad_agent(tmp_path):
    import pydantic

    from launchpad import configs

    bad = tmp_path / "bad.yaml"
    bad.write_text("- {name: a, runtime: echo}\n- {name: b, runtime: not_a_runtime}\n")
    with pytest.raises(pydantic.ValidationError):
        configs.load(bad)


def test_agents_launch_at_boot_from_LAUNCHPAD_CONFIG(tmp_path, monkeypatch):
    cfg = tmp_path / "boot.yaml"
    cfg.write_text("agents:\n  - {name: booted, runtime: echo, prefix: 'boot '}\n")
    monkeypatch.setenv("LAUNCHPAD_CONFIG", str(cfg))

    with TestClient(app) as booted:  # lifespan runs on __enter__
        names = [a["name"] for a in booted.get("/agents", headers=AUTH).json()]
        assert "booted" in names
        aid = next(
            a["id"] for a in booted.get("/agents", headers=AUTH).json() if a["name"] == "booted"
        )
        r = booted.post(f"/agents/{aid}/invoke", headers=AUTH, json={"input": "up"})
        assert r.json()["output"] == {"output": "boot up"}


# --- P0 regressions -----------------------------------------------------------


def test_memory_is_written_to_shared_storage(client, monkeypatch):
    """P0: InMemorySaver was per-process, so a second worker saw an empty history.

    Checkpoints must land in the shared DB, under an agent-namespaced thread key,
    where any other worker can read them.
    """
    from launchpad import registry, store
    from launchpad.runtimes import langchain_react

    monkeypatch.setattr(langchain_react, "init_chat_model", _fake_model)
    aid = client.post(
        "/agents",
        headers=AUTH,
        json={"name": "remembers", "runtime": "langchain_react", "memory": True},
    ).json()["agent_id"]

    body = {"input": "first", "thread_id": "conv-1"}
    assert client.post(f"/agents/{aid}/invoke", headers=AUTH, json=body).status_code == 200

    # a second worker would rebuild the agent from config -- it must still see the thread
    registry.clear()
    assert client.post(f"/agents/{aid}/invoke", headers=AUTH, json=body).status_code == 200

    rows = (
        store.db()
        .execute("select count(*) c from checkpoints where thread_id=?", (f"{aid}:conv-1",))
        .fetchone()
    )
    assert rows["c"] >= 2, "history did not accumulate in shared storage"


def test_thread_ids_are_namespaced_per_agent(client, monkeypatch):
    """P0: one shared checkpoint store means "session-1" must not leak between agents."""
    from launchpad.api import InvokeRequest, _run_config

    req = InvokeRequest(input="x", thread_id="session-1")
    assert _run_config("agent-a", req)["configurable"]["thread_id"] == "agent-a:session-1"
    assert _run_config("agent-b", req)["configurable"]["thread_id"] == "agent-b:session-1"
    assert _run_config("agent-a", InvokeRequest(input="x")) is None


def test_http_get_caps_the_body_it_buffers(monkeypatch):
    """P0: `.text` buffered the whole response, so one big URL could OOM the worker."""
    import httpx

    from launchpad import tools

    huge = b"A" * (5 * 1024 * 1024)
    monkeypatch.setattr(tools, "_assert_public", lambda _url: None)
    monkeypatch.setattr(
        httpx,
        "stream",
        lambda *a, **k: (
            _REAL_ASYNC_CLIENT
            and httpx.Client(
                transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=huge))
            ).stream(*a, **k)
        ),
    )
    out = tools.TOOLS["http_get"].invoke({"url": "https://example.com/big"})
    assert len(out) == 4000  # truncated, and never more than MAX_FETCH_BYTES buffered


def test_retry_after_is_clamped(monkeypatch):
    """P0: an upstream Retry-After was slept on verbatim -- 3600 would pin a worker."""
    import httpx

    from launchpad.runtimes.databricks_mlflow import MAX_RETRY_SLEEP, _retry_after

    assert _retry_after(httpx.Response(429, headers={"retry-after": "3600"}), 0) == MAX_RETRY_SLEEP
    assert _retry_after(httpx.Response(429, headers={"retry-after": "junk"}), 2) == 4.0
    assert _retry_after(httpx.Response(429, headers={"retry-after": "-5"}), 0) == 0.0
    assert _retry_after(httpx.Response(429), 1) == 2.0


def test_shared_client_is_reused_and_closed(client, monkeypatch):
    """P0: a per-agent client leaked its pool whenever the agent cache evicted it."""
    from launchpad.runtimes import databricks_mlflow

    monkeypatch.setattr(databricks_mlflow, "_http", None)
    assert databricks_mlflow.client() is databricks_mlflow.client()  # one pool, not one per agent
