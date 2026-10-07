"""Tool allowlist for agent runtimes.

Configs reference tools **by name**. Nothing from a config is ever imported,
eval'd, or turned into a callable -- that is the injection boundary.
"""

import ipaddress
import socket
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
from langchain_core.tools import BaseTool, tool

TOOLS: dict[str, BaseTool] = {}
MAX_FETCH_BYTES = 200_000  # a tool must not be able to OOM the worker


def register(t: BaseTool) -> BaseTool:
    TOOLS[t.name] = t
    return t


def _assert_public(url: str) -> None:
    """Block SSRF into the platform's own network.

    ponytail: resolves once and checks the literal; a DNS-rebinding attacker can
    still win the race. Move to a pinned-IP transport if tools ever run untrusted.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("only http(s) URLs are allowed")
    ip = ipaddress.ip_address(socket.gethostbyname(parts.hostname))
    if not ip.is_global:
        raise ValueError(f"refusing to fetch a non-public address ({ip})")


@register
@tool
def http_get(url: str) -> str:
    """Fetch a public http(s) URL and return the first 4000 characters of the body."""
    _assert_public(url)
    # Streamed and capped: `httpx.get(...).text` would buffer the whole response
    # first, so one large URL could exhaust memory.
    body = bytearray()
    with httpx.stream("GET", url, timeout=10, follow_redirects=False) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            body += chunk
            if len(body) >= MAX_FETCH_BYTES:
                break
    return body[:MAX_FETCH_BYTES].decode("utf-8", errors="replace")[:4000]


@register
@tool
def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()
