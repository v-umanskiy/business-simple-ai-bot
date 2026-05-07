import json
from typing import Any

from src.ai.model_provider import get_model_provider


def parse_mode_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_decision(payload: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    mode = str(payload.get("mode") or "")
    if mode == "answer":
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {"mode": "answer", "answer": answer.strip()}
        return {}

    if mode == "request_sections":
        values = payload.get("sections")
        if not isinstance(values, list):
            return {"mode": "request_sections", "sections": []}
        sections = [str(v) for v in values if isinstance(v, str) and str(v) in allowed]
        return {"mode": "request_sections", "sections": sections}

    return {}


async def decide_answer_or_sections(
    *,
    message_text: str,
    base_system_prompt: str,
    section_names: list[str],
) -> dict[str, Any]:
    allowed = set(section_names)
    selector_system = (
        f"{base_system_prompt}\n\n"
        "You must decide one of two actions and return ONLY JSON.\n"
        "Action 1 (direct answer): {\"mode\":\"answer\",\"answer\":\"...\"}\n"
        "Action 2 (request sections): {\"mode\":\"request_sections\",\"sections\":[\"NAME\",\"NAME\"]}\n"
        "If you need business details, request sections.\n"
        "Use only section names from the allowed list.\n\n"
        f"Allowed section names: {', '.join(section_names)}"
    )

    provider = get_model_provider()
    raw = await provider.generate_reply(
        prompt=message_text,
        system_prompt=selector_system,
        history=[],
    )
    decision = _normalize_decision(parse_mode_response(raw), allowed)
    if decision:
        return decision

    repair_system = (
        "Return ONLY valid JSON in one of these exact shapes:\n"
        "{\"mode\":\"answer\",\"answer\":\"...\"}\n"
        "{\"mode\":\"request_sections\",\"sections\":[\"NAME\",\"NAME\"]}\n"
        f"Allowed section names: {', '.join(section_names)}"
    )
    repair_prompt = (
        "The previous output was invalid. Reformat it to valid JSON only.\n\n"
        f"Previous output:\n{raw}"
    )
    repaired_raw = await provider.generate_reply(
        prompt=repair_prompt,
        system_prompt=repair_system,
        history=[],
    )
    repaired_decision = _normalize_decision(parse_mode_response(repaired_raw), allowed)
    if repaired_decision:
        return repaired_decision

    return {"mode": "request_sections", "sections": []}
