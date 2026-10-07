"""SQLite persistence. Agents are content-addressed: same config -> same id.

ponytail: stdlib sqlite3, no ORM. Swap for Postgres when there is >1 replica.
"""

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid

from launchpad.config import AgentConfig, parse

SCHEMA = """
create table if not exists agents (
    id text primary key,
    name text not null,
    runtime text not null,
    config text not null,
    created_at real not null
);
create table if not exists runs (
    id text primary key,
    agent_id text not null,
    input text,
    output text,
    error text,
    latency_ms real,
    tokens integer,
    created_at real not null
);
create index if not exists runs_agent on runs(agent_id, created_at);
"""


# A sqlite3 connection is not safe to share between threads -- async handlers
# offload to a threadpool, and concurrent use of one connection raises
# "bad parameter or other API misuse". One connection per thread instead; WAL
# plus busy_timeout is what makes them safe against each other.
_local = threading.local()


def _connect(path: str) -> sqlite3.Connection:
    # isolation_level=None (autocommit) matters for multi-worker correctness: the
    # default leaves a read transaction open on the connection, pinning it to the
    # snapshot of its first read, so it would never see another worker's writes.
    conn = sqlite3.connect(path, check_same_thread=False, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # busy_timeout FIRST -- switching journal modes takes an exclusive lock, and
    # without a timeout already set, concurrent workers fail instead of waiting.
    conn.execute("pragma busy_timeout=15000")
    # WAL lets the other workers read while one writes. It is a persistent property
    # of the file, so whichever worker wins the race sets it for everyone -- and the
    # losers must not die: journal_mode is the one pragma that ignores busy_timeout
    # and returns BUSY immediately while another connection is open.
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("pragma journal_mode=WAL")
    conn.execute("pragma synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def db() -> sqlite3.Connection:
    path = os.environ.get("LAUNCHPAD_DB", "launchpad.db")
    cached = getattr(_local, "conn", None)
    if cached is None or cached[0] != path:
        cached = (path, _connect(path))
        _local.conn = cached
    return cached[1]


def agent_id(cfg: AgentConfig) -> str:
    """Content address: idempotent creates, and a stable cache key for built agents."""
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_agent(cfg: AgentConfig) -> str:
    aid = agent_id(cfg)
    db().execute(
        "insert or ignore into agents values (?,?,?,?,?)",
        (aid, cfg.name, cfg.runtime, cfg.model_dump_json(), time.time()),
    )
    return aid


def load_agent(aid: str) -> AgentConfig | None:
    row = db().execute("select config from agents where id=?", (aid,)).fetchone()
    return parse(row["config"]) if row else None


def list_agents() -> list[dict]:
    rows = (
        db()
        .execute("select id, name, runtime, created_at from agents order by created_at desc")
        .fetchall()
    )
    return [dict(r) for r in rows]


def record_run(
    agent_id: str,
    input_: object,
    output: object = None,
    error: str | None = None,
    latency_ms: float = 0.0,
    tokens: int | None = None,
) -> str:
    run_id = uuid.uuid4().hex
    db().execute(
        "insert into runs values (?,?,?,?,?,?,?,?)",
        (
            run_id,
            agent_id,
            json.dumps(input_, default=str),
            json.dumps(output, default=str) if output is not None else None,
            error,
            latency_ms,
            tokens,
            time.time(),
        ),
    )
    return run_id


def load_run(run_id: str) -> dict | None:
    row = db().execute("select * from runs where id=?", (run_id,)).fetchone()
    if not row:
        return None
    out = dict(row)
    for k in ("input", "output"):
        out[k] = json.loads(out[k]) if out[k] else None
    return out
