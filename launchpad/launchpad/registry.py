"""The whole platform: a dict from runtime name to a builder function.

Every runtime returns a LangChain `Runnable` (or anything with the same two
methods), so invoke/stream/async/callbacks come for free and there is no
adapter layer. Adding a runtime is one decorated function.
"""

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from launchpad.config import AgentConfig


class Runnable(Protocol):
    async def ainvoke(self, input: Any, config: dict | None = None) -> Any: ...
    def astream(self, input: Any, config: dict | None = None) -> AsyncIterator[Any]: ...


Builder = Callable[[Any], Runnable]
RUNTIMES: dict[str, Builder] = {}


class UnknownRuntime(KeyError):
    pass


def runtime(name: str) -> Callable[[Builder], Builder]:
    def deco(fn: Builder) -> Builder:
        RUNTIMES[name] = fn
        return fn

    return deco


def build(cfg: AgentConfig) -> Runnable:
    if cfg.runtime not in RUNTIMES:
        raise UnknownRuntime(f"unknown runtime {cfg.runtime!r}; available: {sorted(RUNTIMES)}")
    return RUNTIMES[cfg.runtime](cfg)


# Configs are content-addressed, so a built agent can never go stale: the id IS
# the config hash. ponytail: unbounded, but bounded in practice by the number of
# distinct configs; add an LRU if someone starts generating agents per request.
_built: dict[str, Runnable] = {}


def get_agent(agent_id: str, cfg: AgentConfig) -> Runnable:
    """Built once per config, reused across requests."""
    agent = _built.get(agent_id)
    if agent is None:
        agent = _built[agent_id] = build(cfg)
    return agent


def clear() -> None:
    """Drop built agents -- what a fresh worker process would start with."""
    _built.clear()
