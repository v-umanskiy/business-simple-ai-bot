import json
import os

import boto3

_sm = boto3.client("secretsmanager")
_cache: dict[str, str | None] = {}
_payload: dict | None = None



def _secret_id() -> str:
    return os.getenv("SECRETS_ARN") or "BUSINESS_SIMPLE_AI_BOT_SECRET"



def _load_payload() -> dict:
    global _payload
    if _payload is not None:
        return _payload

    response = _sm.get_secret_value(SecretId=_secret_id())
    secret_string = response.get("SecretString", "")
    _payload = json.loads(secret_string) if secret_string else {}
    return _payload



def get_secret(key: str) -> str | None:
    if key in _cache:
        return _cache[key]

    payload = _load_payload()
    value = payload.get(key)
    _cache[key] = value
    return value
