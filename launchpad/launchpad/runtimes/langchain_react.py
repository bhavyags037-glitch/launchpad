"""LangGraph ReAct agent, built from config.

Do not hand-roll the ReAct loop -- `create_agent` is the maintained one.
"""

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda

from launchpad import checkpoints
from launchpad.config import LangChainReactConfig
from launchpad.registry import runtime
from launchpad.tools import TOOLS


def _as_messages(x: Any) -> dict:
    if isinstance(x, str):
        return {"messages": [{"role": "user", "content": x}]}
    if isinstance(x, dict) and "messages" in x:
        return x
    raise ValueError("input must be a string or {'messages': [...]}")


def _final(state: dict) -> dict:
    """Last message out of either a full graph state or a streamed node update.

    `ainvoke` hands back {"messages": [...]}; `astream` hands back one
    {node_name: {"messages": [...]}} update per step.
    """
    messages = state.get("messages")
    if messages is None:
        messages = [
            m
            for update in state.values()
            if isinstance(update, dict)
            for m in update.get("messages", [])
        ]
    return {"output": messages[-1].content if messages else ""}


@runtime("langchain_react")
def build(cfg: LangChainReactConfig) -> Runnable:
    unknown = set(cfg.tools) - set(TOOLS)
    if unknown:
        raise ValueError(f"unknown tools {sorted(unknown)}; allowed: {sorted(TOOLS)}")

    kwargs: dict[str, Any] = {"max_tokens": cfg.model.max_tokens}
    if cfg.model.temperature is not None:
        kwargs["temperature"] = cfg.model.temperature
    if cfg.model.base_url:
        kwargs["base_url"] = cfg.model.base_url
    model = init_chat_model(cfg.model.name, model_provider=cfg.model.provider, **kwargs)

    agent = create_agent(
        model,
        [TOOLS[name] for name in cfg.tools],
        system_prompt=cfg.system_prompt,
        checkpointer=checkpoints.saver() if cfg.memory else None,
    )
    # ponytail: streaming yields one chunk per graph step (tool result, then answer),
    # not per token. Swap for `astream_events` if per-token output is ever asked for.
    return (RunnableLambda(_as_messages) | agent | RunnableLambda(_final)).with_config(
        {"recursion_limit": 2 * cfg.limits.max_iterations + 1}
    )
