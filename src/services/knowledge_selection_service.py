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


def _normalize_decision(
    payload: dict[str, Any],
    allowed: set[int],
    allowed_image_ids: set[str],
) -> dict[str, Any]:
    def _normalize_image_ids(raw_values: Any, allowed_image_ids: set[str]) -> list[str]:
        if not isinstance(raw_values, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for value in raw_values:
            if not isinstance(value, str):
                continue
            image_id = value.strip()
            if not image_id or image_id not in allowed_image_ids or image_id in seen:
                continue
            seen.add(image_id)
            normalized.append(image_id)
            if len(normalized) >= 3:
                break
        return normalized

    mode = str(payload.get("mode") or "")
    if mode == "answer":
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            image_ids = _normalize_image_ids(payload.get("image_ids"), allowed_image_ids)
            return {"mode": "answer", "answer": answer.strip(), "image_ids": image_ids}
        return {}

    if mode == "handoff":
        lang = payload.get("lang")
        if isinstance(lang, str) and lang.strip():
            return {"mode": "handoff", "lang": lang.strip().lower()}
        return {"mode": "handoff", "lang": "en"}

    if mode == "request_sections":
        values = payload.get("sections")
        if not isinstance(values, list):
            return {"mode": "request_sections", "sections": []}
        sections: list[int] = []
        seen: set[int] = set()
        for value in values:
            number: int | None = None
            if isinstance(value, int):
                number = value
            elif isinstance(value, str) and value.strip().isdigit():
                number = int(value.strip())
            if number is None or number not in allowed or number in seen:
                continue
            seen.add(number)
            sections.append(number)
        return {"mode": "request_sections", "sections": sections}

    return {}


async def decide_answer_or_sections(
    *,
    message_text: str,
    base_system_prompt: str,
    section_options: list[dict],
    image_ids: list[str],
    history: list[dict] | None = None,
) -> dict[str, Any]:
    allowed = {
        option["number"]
        for option in section_options
        if isinstance(option.get("number"), int) and isinstance(option.get("title"), str)
    }
    allowed_images = set(image_ids)
    section_lines = "\n".join(
        f"{option['number']}. {option['title']}"
        for option in section_options
        if isinstance(option.get("number"), int) and isinstance(option.get("title"), str)
    )
    selector_system = (
        f"{base_system_prompt}\n\n"
        "You must decide one of three actions and return ONLY JSON.\n"
        "Action 1 (direct answer): {\"mode\":\"answer\",\"answer\":\"...\",\"image_ids\":[\"ID\"]}\n"
        "Action 2 (request sections): {\"mode\":\"request_sections\",\"sections\":[1,2]}\n"
        "Action 3 (handoff): {\"mode\":\"handoff\",\"lang\":\"...\"}\n"
        "If you need business details, request sections.\n"
        "For a direct answer to a user always use the same language that as the user used in their question."
        "Do NOT use markdown (e.g. ** for bold). You can use only plain text options if needed, e.g. numbers, dashes, capitalizing."
        "Use handoff only when the user explicitly asks for a human operator or escalation.\n"
        "For handoff, set lang to the user's language code (for example: de, en, es).\n"
        "If relevant images are available, include image_ids in answer mode.\n"
        "Use only section numbers from the allowed list.\n\n"
        f"Allowed sections:\n{section_lines}\n"
        f"Allowed image IDs (optional for answer mode, max 3): {', '.join(image_ids)}"
    )

    provider = get_model_provider()
    raw = await provider.generate_reply(
        prompt=message_text,
        system_prompt=selector_system,
        history=history or [],
    )
    decision = _normalize_decision(parse_mode_response(raw), allowed, allowed_images)
    if decision:
        return decision

    repair_system = (
        "Return ONLY valid JSON in one of these exact shapes:\n"
        "{\"mode\":\"answer\",\"answer\":\"...\",\"image_ids\":[\"ID\"]}\n"
        "{\"mode\":\"request_sections\",\"sections\":[1,2]}\n"
        "{\"mode\":\"handoff\",\"lang\":\"...\"}\n"
        f"Allowed sections:\n{section_lines}\n"
        f"Allowed image IDs (optional for answer mode, max 3): {', '.join(image_ids)}"
    )
    repair_prompt = (
        "The previous output was invalid. Reformat it to valid JSON only.\n\n"
        f"Previous output:\n{raw}"
    )
    repaired_raw = await provider.generate_reply(
        prompt=repair_prompt,
        system_prompt=repair_system,
        history=history or [],
    )
    repaired_decision = _normalize_decision(parse_mode_response(repaired_raw), allowed, allowed_images)
    if repaired_decision:
        return repaired_decision

    return {"mode": "answer", "answer": "Sorry, I could not process that request right now."}
