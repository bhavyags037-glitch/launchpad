"""Trivial runtime: proves the platform works with zero LLM dependencies."""

from collections.abc import AsyncIterator
from typing import Any

from launchpad.config import EchoConfig
from launchpad.registry import runtime


class _Echo:
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    async def ainvoke(self, input: Any, config: dict | None = None) -> Any:
        return {"output": f"{self.prefix}{input}"}

    async def astream(self, input: Any, config: dict | None = None) -> AsyncIterator[Any]:
        for word in f"{self.prefix}{input}".split():
            yield {"output": word}


@runtime("echo")
def build(cfg: EchoConfig) -> _Echo:
    return _Echo(cfg.prefix)
