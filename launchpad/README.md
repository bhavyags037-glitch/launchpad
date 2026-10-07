# Launchpad

A config-driven agent platform. You declare an agent; the platform builds, serves,
limits, and records it. No per-agent code, no redeploy.

Three runtimes today — `echo`, `langchain_react`, `databricks_mlflow` — all behind
one `POST /agents/{id}/invoke` contract.

---

## 1. Install

```bash
git clone <your-repo-url> launchpad && cd launchpad
uv sync --extra langchain --group dev
```

`uv sync` alone gets you `echo` + `databricks_mlflow`. Add `--extra langchain` for
the LangChain/LangGraph runtime.

## 2. Set the environment

```bash
export LAUNCHPAD_API_KEYS=dev-key        # REQUIRED — auth fails closed without it
export LAUNCHPAD_API_KEY=dev-key         # what the CLI sends
export ANTHROPIC_API_KEY=sk-ant-...      # or OPENAI_API_KEY, per your config
```

Copy `.env.example` if you prefer a file. Secrets are read from the environment and
are never written into a stored config.

## 3. Declare your agents

`agents.yaml` — the same schema the HTTP API accepts. YAML or JSON, a list or an
`agents:` key:

```yaml
agents:
  - name: greeter                 # no LLM needed — good first smoke test
    runtime: echo
    prefix: "hi "

  - name: researcher
    runtime: langchain_react
    model:
      provider: anthropic
      name: claude-opus-5
    tools: [http_get, utc_now]    # allowlisted by name
    system_prompt: Answer with sources. Be terse.
    memory: true                  # per-thread_id conversation history
    limits:
      timeout_s: 120
      max_iterations: 8
```

## 4. Run it

```bash
LAUNCHPAD_CONFIG=agents.yaml uv run uvicorn launchpad.api:app --reload
# INFO:  launched greeter [echo] -> 3f6a05fa9d659f45
# INFO:  launched researcher [langchain_react] -> 7f0da264bb0486bd
```

Every agent in the file is validated, built, and serving before the first request.
A bad config kills the boot instead of failing later at invoke time.

For more than one worker (memory is shared via SQLite, so this is safe):

```bash
LAUNCHPAD_CONFIG=agents.yaml uv run uvicorn launchpad.api:app --workers 4
```

---

## CLI

```bash
uv run launchpad ls                                  # what is running
uv run launchpad apply agents.yaml                   # declare agents on a running platform
uv run launchpad invoke greeter "world"              # by name or id
uv run launchpad invoke researcher "what time is it?" --thread-id session-1
uv run python -m launchpad_cli ls                    # same thing, no entry point needed
```

| Flag | Env | Default |
|---|---|---|
| `--url` | `LAUNCHPAD_URL` | `http://localhost:8000` |
| `--key` | `LAUNCHPAD_API_KEY` | first entry of `LAUNCHPAD_API_KEYS` |
| `--thread-id` | — | none (no memory) |

`apply` is idempotent: the agent id is a hash of the config, so re-applying an
unchanged file returns the same ids. Configs are validated locally before anything
leaves the machine. It does not yet delete agents removed from the file.

Point it at a remote platform:

```bash
uv run launchpad --url https://agents.internal --key "$PROD_KEY" ls
```

## HTTP API

The CLI is a thin client over this.

```bash
curl -XPOST localhost:8000/agents -H 'X-API-Key: dev-key' -H 'content-type: application/json' -d '{
  "name": "greeter", "runtime": "echo", "prefix": "hi "}'
# -> {"agent_id":"3f6a05fa9d659f45","name":"greeter","runtime":"echo"}

curl -XPOST localhost:8000/agents/3f6a05fa9d659f45/invoke -H 'X-API-Key: dev-key' \
  -H 'content-type: application/json' -d '{"input":"world"}'
# -> {"run_id":"bf01...","output":{"output":"hi world"}}

# streaming (SSE)
curl -sN -XPOST localhost:8000/agents/<id>/invoke -H 'X-API-Key: dev-key' \
  -H 'content-type: application/json' -d '{"input":"go","stream":true}'
```

| Endpoint | Purpose |
|---|---|
| `POST /agents` | validate + build + persist a config |
| `GET /agents`, `GET /agents/{id}` | list / read |
| `POST /agents/{id}/invoke` | `{input, stream?, thread_id?}` |
| `GET /runs/{run_id}` | input, output, error, latency |
| `GET /healthz` | registered runtimes |

Interactive docs at `http://localhost:8000/docs`.

---

## Config reference

Shared by every runtime:

| Field | Default | Notes |
|---|---|---|
| `name` | required | 1–128 chars |
| `runtime` | required | `echo` \| `langchain_react` \| `databricks_mlflow` |
| `limits.timeout_s` | `60` | 0 < t ≤ 600 |
| `limits.max_iterations` | `10` | 0 < n ≤ 100 |

**`langchain_react`**

| Field | Default | Notes |
|---|---|---|
| `model.provider` | `anthropic` | any `init_chat_model` provider |
| `model.name` | `claude-opus-5` | |
| `model.temperature` | unset | leave unset for current Claude models — they reject it with a 400 |
| `model.max_tokens` | `16000` | |
| `model.base_url` | unset | vLLM / LiteLLM / a private gateway |
| `tools` | `[]` | `http_get`, `utc_now` — allowlisted names only |
| `system_prompt` | none | |
| `memory` | `false` | enables `thread_id` history |

**`databricks_mlflow`**

| Field | Default | Notes |
|---|---|---|
| `endpoint` | required | serving endpoint name |
| `mode` | `chat` | `chat` for ChatAgent/OpenAI shapes, `raw` for pyfunc |
| `retries` | `2` | 0–5, backoff on 429/502/503/504 |

## Environment

| Var | Meaning |
|---|---|
| `LAUNCHPAD_API_KEYS` | Comma-separated API keys. **Required** — auth fails closed. |
| `LAUNCHPAD_API_KEY` / `LAUNCHPAD_URL` | What the CLI sends / where it points. |
| `LAUNCHPAD_CONFIG` | YAML/JSON file of agents to launch at boot. Optional. |
| `LAUNCHPAD_DB` | SQLite path (default `launchpad.db`). Holds agents, runs, and memory. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | For `langchain_react`, per `model.provider`. |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | For `databricks_mlflow`. |

## Troubleshooting

| Symptom | Cause |
|---|---|
| `500 LAUNCHPAD_API_KEYS is not configured` | Auth fails closed by design. Set it. |
| `401 invalid or missing X-API-Key` | CLI has no key — set `LAUNCHPAD_API_KEY`. |
| `400 cannot build agent: unknown tools [...]` | Tools are allowlisted by name; the error lists what's allowed. |
| `400 cannot build agent: ...` | Config is invalid or a provider key is missing. Fails at create, not invoke. |
| `504` | Exceeded `limits.timeout_s`. |
| `502` / `429` | Upstream model or endpoint failed. `GET /runs/{id}` has the recorded error. |
| Agent forgets mid-conversation | Set `memory: true` *and* pass `--thread-id`. |

## Tests

```bash
uv run pytest        # 23 tests, no network, no API keys
uv run ruff check .
```

## How it works

Every runtime is a function `AgentConfig -> Runnable`, registered in a dict:

```python
@runtime("langchain_react")
def build(cfg: LangChainReactConfig) -> Runnable:
    ...
```

Because a LangGraph agent, a chat model, and an HTTP wrapper are all `Runnable`s,
invoke/stream/async/callbacks come free — no adapter layer, no base class, no
factory. Agents are content-addressed by config hash, so a built agent can never
go stale.

## For coding agents

`.claude/skills/launchpad-agent/SKILL.md` is a Claude Code skill covering how to
scaffold a project, declare a LangChain agent, set the environment, boot, and
verify — plus the failure modes worth knowing. It is auto-discovered in this repo;
copy the directory into another project to carry it along.

See `PLAN.md` for what was deliberately left out and when to add it.
