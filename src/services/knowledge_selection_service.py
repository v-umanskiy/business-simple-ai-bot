import json
from typing import Any

from src.ai.model_provider import get_model_provider


def parse_mode_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


async def decide_answer_or_sections(
    *,
    message_text: str,
    base_system_prompt: str,
    section_names: list[str],
) -> dict[str, Any]:
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
    return parse_mode_response(raw)
