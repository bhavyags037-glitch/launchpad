"""Conversation memory shared by every worker process.

LangGraph's `InMemorySaver` lives in one process, so with more than one uvicorn
worker a conversation loses its history whenever a turn lands on a different
worker. One SQLite-backed saver per process, all pointing at the same file,
fixes that -- the same trade the rest of the platform already makes.

ponytail: SQLite + WAL holds for a single box. Swap in `AsyncPostgresSaver` (same
interface, one line) when the platform runs on more than one machine.
"""

import contextlib
import os
import sqlite3

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

_saver: AsyncSqliteSaver | None = None
_conn: aiosqlite.Connection | None = None
_users = 0  # apps sharing this store; tests embed more than one in a process


async def start() -> None:
    global _saver, _conn, _users
    _users += 1
    if _saver is not None:
        return
    _conn = await aiosqlite.connect(
        os.environ.get("LAUNCHPAD_DB", "launchpad.db"), isolation_level=None
    )
    # busy_timeout before anything that takes a lock: several workers run this
    # concurrently at boot, and the WAL switch plus setup() DDL both need one.
    await _conn.execute("pragma busy_timeout=30000")
    # Persistent on the file, and it ignores busy_timeout: whichever worker wins
    # sets it for all of them, and the rest must not fail their startup over it.
    with contextlib.suppress(sqlite3.OperationalError):
        await _conn.execute("pragma journal_mode=WAL")
    _saver = AsyncSqliteSaver(_conn)
    await _saver.setup()


async def close() -> None:
    global _saver, _conn, _users
    _users = max(0, _users - 1)
    if _users or _conn is None:
        return
    await _conn.close()
    _saver, _conn = None, None


def saver() -> AsyncSqliteSaver:
    if _saver is None:
        raise RuntimeError("checkpoint store not started; agents with memory need the app running")
    return _saver
