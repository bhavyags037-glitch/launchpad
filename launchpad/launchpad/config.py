"""Agent configs. One model per runtime, discriminated on `runtime`.

A bad config is rejected at POST /agents with a field path, never at invoke time.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class Limits(BaseModel):
    timeout_s: float = Field(default=60, gt=0, le=600)
    max_iterations: int = Field(default=10, gt=0, le=100)


class BaseAgentConfig(BaseModel, frozen=True):
    name: str = Field(min_length=1, max_length=128)
    limits: Limits = Limits()


class EchoConfig(BaseAgentConfig):
    """Trivial runtime. Exists so the platform is testable with zero LLM deps."""

    runtime: Literal["echo"]
    prefix: str = ""


class ModelSpec(BaseModel):
    provider: str = "anthropic"
    name: str = "claude-opus-5"
    # ponytail: left unset by default -- current Claude models reject `temperature` with a 400.
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=16000, gt=0)
    # self-hosted or gatewayed models (vLLM, LiteLLM, Ollama, a private proxy)
    base_url: str | None = None


class LangChainReactConfig(BaseAgentConfig):
    runtime: Literal["langchain_react"]
    model: ModelSpec = ModelSpec()
    tools: tuple[str, ...] = ()  # names only, resolved against an allowlist at create time
    system_prompt: str | None = None
    memory: bool = False  # per-thread_id checkpointing


class DatabricksConfig(BaseAgentConfig):
    """A Databricks Model Serving endpoint (MLflow-deployed agent or model).

    Credentials never live in the config -- they come from DATABRICKS_HOST /
    DATABRICKS_TOKEN in the platform's environment.
    """

    runtime: Literal["databricks_mlflow"]
    endpoint: str = Field(min_length=1)
    # "chat": ChatAgent / OpenAI-shaped endpoints.  "raw": mlflow pyfunc `inputs` payloads.
    mode: Literal["chat", "raw"] = "chat"
    retries: int = Field(default=2, ge=0, le=5)


AgentConfig = Annotated[
    EchoConfig | LangChainReactConfig | DatabricksConfig, Field(discriminator="runtime")
]
_adapter: TypeAdapter[AgentConfig] = TypeAdapter(AgentConfig)


def parse(raw: str | bytes) -> AgentConfig:
    return _adapter.validate_json(raw)


def parse_obj(data: object) -> AgentConfig:
    return _adapter.validate_python(data)
