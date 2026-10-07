"""API-key auth. Fails closed: no keys configured means no requests served."""

import hmac
import os

from fastapi import Header, HTTPException


def require_key(x_api_key: str = Header(default="")) -> None:
    keys = [k.strip() for k in os.environ.get("LAUNCHPAD_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        raise HTTPException(500, "LAUNCHPAD_API_KEYS is not configured")
    if not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(401, "invalid or missing X-API-Key")
