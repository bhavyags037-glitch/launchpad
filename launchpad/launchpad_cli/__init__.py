"""Launchpad CLI -- a config-driven client for a running platform.

Deliberately a separate package from `launchpad`: it is an HTTP client, so it
installs and ships without FastAPI, langchain, or the server's deps.

launchpad apply agents.yaml     # declare agents
launchpad ls                    # what is running
launchpad invoke researcher "what time is it?"
"""

import argparse
import json
import os
import sys

import httpx

from launchpad import configs


def _client(args: argparse.Namespace) -> httpx.Client:
    key = args.key or os.environ.get("LAUNCHPAD_API_KEYS", "").split(",")[0].strip()
    return httpx.Client(base_url=args.url, headers={"X-API-Key": key}, timeout=300)


def _resolve(http: httpx.Client, ref: str) -> str:
    """Accept an agent id or a name."""
    agents = http.get("/agents").raise_for_status().json()
    match = [a["id"] for a in agents if ref in (a["id"], a["name"])]
    if not match:
        sys.exit(f"no agent {ref!r}; have: {', '.join(a['name'] for a in agents) or '(none)'}")
    return match[0]


def main() -> None:
    parser = argparse.ArgumentParser("launchpad")
    parser.add_argument("--url", default=os.environ.get("LAUNCHPAD_URL", "http://localhost:8000"))
    parser.add_argument("--key", default=os.environ.get("LAUNCHPAD_API_KEY"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls")
    p_apply = sub.add_parser("apply")
    p_apply.add_argument("file")
    p_invoke = sub.add_parser("invoke")
    p_invoke.add_argument("agent")
    p_invoke.add_argument("input")
    p_invoke.add_argument("--thread-id")
    args = parser.parse_args()

    try:
        _run(args)
    except httpx.HTTPStatusError as e:  # a CLI reports the problem, not a stack trace
        sys.exit(f"launchpad: {args.url} returned {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        sys.exit(f"launchpad: cannot reach {args.url}: {e}")
    except (OSError, ValueError) as e:
        sys.exit(f"launchpad: {e}")


def _run(args: argparse.Namespace) -> None:
    with _client(args) as http:
        if args.cmd == "ls":
            for a in http.get("/agents").raise_for_status().json():
                print(f"{a['id']}  {a['runtime']:<18} {a['name']}")

        elif args.cmd == "apply":
            for cfg in configs.load(args.file):  # validated locally before it leaves the machine
                r = http.post("/agents", json=cfg.model_dump(mode="json"))
                if r.is_error:
                    sys.exit(f"{cfg.name}: {r.text}")
                print(f"{r.json()['agent_id']}  {cfg.runtime:<18} {cfg.name}")

        elif args.cmd == "invoke":
            body = {"input": args.input, "thread_id": args.thread_id}
            r = http.post(f"/agents/{_resolve(http, args.agent)}/invoke", json=body)
            if r.is_error:
                sys.exit(r.text)
            print(json.dumps(r.json()["output"], indent=2, default=str))
