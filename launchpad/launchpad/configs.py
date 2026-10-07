"""Load agent configs from a YAML or JSON file.

The file is the same schema the API accepts -- one document, a list, or a
top-level `agents:` key. YAML is a superset of JSON, so both just work.
"""

from pathlib import Path

import yaml

from launchpad.config import AgentConfig, parse_obj


def load(path: str | Path) -> list[AgentConfig]:
    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict):
        raw = raw.get("agents", raw)
    return [parse_obj(item) for item in (raw if isinstance(raw, list) else [raw])]
