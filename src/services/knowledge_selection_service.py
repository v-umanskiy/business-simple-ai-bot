import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from src.ai.model_provider import get_model_provider
from src.utils.log_utils import chars

logger = logging.getLogger(__name__)


class BotDecisionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | None
    image_ids: list[str]
    sections: list[int]
    lang: str | None


class BotDecisionWithSectionRequest(BotDecisionBase):
    mode: Literal["answer", "request_sections", "handoff"]


class BotDecisionFinal(BotDecisionBase):
    mode: Literal["answer", "handoff"]


def parse_mode_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_mode_response_candidates(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    candidates: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    offset = 0

    while offset < len(text):
        object_start = text.find("{", offset)
        if object_start < 0:
            break
        try:
            payload, object_end = decoder.raw_decode(text, object_start)
        except json.JSONDecodeError:
            offset = object_start + 1
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
        offset = object_end

    return candidates


def _normalize_decision(
    payload: dict[str, Any],
    allowed: set[int],
    allowed_image_ids: set[str],
    allow_section_requests: bool,
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

    if mode == "request_sections" and allow_section_requests:
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


def _select_decision(
    raw: str,
    allowed: set[int],
    allowed_image_ids: set[str],
    allow_section_requests: bool,
) -> dict[str, Any]:
    model = _decision_model(allow_section_requests)
    candidates = []
    for payload in _parse_mode_response_candidates(raw):
        try:
            typed_payload = model.model_validate(payload)
        except ValidationError:
            continue

        decision = _normalize_decision(
            typed_payload.model_dump(),
            allowed,
            allowed_image_ids,
            allow_section_requests,
        )
        if decision:
            candidates.append(decision)

    if len(candidates) == 1:
        return candidates[0]

    if allow_section_requests:
        requested_sections: list[int] = []
        seen: set[int] = set()
        for decision in candidates:
            if decision.get("mode") != "request_sections":
                continue
            for section in decision.get("sections") or []:
                if section not in seen:
                    seen.add(section)
                    requested_sections.append(section)
        if requested_sections:
            return {"mode": "request_sections", "sections": requested_sections}

    return {}


def _format_section_options(section_options: list[dict]) -> str:
    return "\n".join(
        f"{option['number']}. {option['title']}"
        for option in section_options
        if isinstance(option.get("number"), int) and isinstance(option.get("title"), str)
    )


def _format_provided_sections(sections: list[dict]) -> str:
    return "\n\n".join(f"[{section['title']}]\n{section['text']}" for section in sections)


def _json_shapes(allow_section_requests: bool) -> list[str]:
    shapes = [
        '{"mode":"answer","answer":"...","image_ids":["ID"],"sections":[],"lang":null}',
    ]
    if allow_section_requests:
        shapes.append(
            '{"mode":"request_sections","answer":null,"image_ids":[],"sections":[1,2],"lang":null}'
        )
    shapes.append('{"mode":"handoff","answer":null,"image_ids":[],"sections":[],"lang":"..."}')
    return shapes


def _json_shapes_text(allow_section_requests: bool) -> str:
    return "\n".join(_json_shapes(allow_section_requests))


def _decision_model(allow_section_requests: bool) -> type[BotDecisionBase]:
    return BotDecisionWithSectionRequest if allow_section_requests else BotDecisionFinal


def _strip_schema_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_schema_titles(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_strip_schema_titles(item) for item in value]
    return value


def _decision_response_schema(allow_section_requests: bool) -> dict[str, Any]:
    return _strip_schema_titles(_decision_model(allow_section_requests).model_json_schema())


def _action_instructions(allow_section_requests: bool) -> str:
    names = (
        ["direct answer", "request sections", "handoff"]
        if allow_section_requests
        else ["direct answer", "handoff"]
    )
    return "\n".join(
        f"Action {number} ({name}): {shape}"
        for number, (name, shape) in enumerate(zip(names, _json_shapes(allow_section_requests)), start=1)
    )


async def decide_next_action(
    *,
    message_text: str,
    base_system_prompt: str,
    section_options: list[dict],
    image_ids: list[str],
    history: list[dict] | None = None,
    provided_sections: list[dict] | None = None,
    allow_section_requests: bool = True,
) -> dict[str, Any]:
    allowed = {
        option["number"]
        for option in section_options
        if isinstance(option.get("number"), int) and isinstance(option.get("title"), str)
    }
    allowed_images = set(image_ids)
    section_lines = _format_section_options(section_options)
    provided_sections_text = _format_provided_sections(provided_sections or [])
    section_request_instruction = (
        "If you need business details that are not in the provided sections, request sections.\n"
        if allow_section_requests
        else "Use the provided business knowledge sections as the factual source. If they are insufficient, answer with that limitation or choose handoff.\n"
    )
    selector_system = (
        f"{base_system_prompt}\n\n"
        "You are a business purpose bot, consulting users based on the provided business knowledge. "
        "If a user asks an unrelated question, briefly respond that you only answer regarding the business. "
        "You must decide the next action and return ONLY JSON.\n"
        f"{_action_instructions(allow_section_requests)}\n"
        f"{section_request_instruction}"
        "Reply in the same language as the user. To determine the response language, mind the user_prompt and history. "
        "Do NOT use markdown (e.g. ** for bold). You can use only plain text options if needed, e.g. numbers, dashes, capitalizing. "
        "Use handoff only when the user explicitly asks for a human operator or escalation.\n"
        "For handoff, set lang to the user's language code (for example: de, en, es).\n"
        "If relevant images are available, include image_ids in answer mode.\n"
        "Use only section numbers from the allowed list.\n\n"
        f"Provided knowledge sections:\n{provided_sections_text}\n\n"
        f"Allowed additional sections:\n{section_lines}\n"
        f"Allowed image IDs (optional for answer mode, max 3): {', '.join(image_ids)}"
    )

    provider = get_model_provider()
    logger.info(
        "LLM selector prompt system_prompt_chars=%s history_turns=%s user_prompt_chars=%s",
        chars(selector_system),
        len(history or []),
        chars(message_text),
    )
    raw = await provider.generate_reply(
        prompt=message_text,
        system_prompt=selector_system,
        history=history or [],
        response_schema=_decision_response_schema(allow_section_requests),
    )
    decision = _select_decision(raw, allowed, allowed_images, allow_section_requests)
    if decision:
        return decision

    repair_system = (
        "Return ONLY valid JSON in one of these exact shapes:\n"
        f"{_json_shapes_text(allow_section_requests)}\n"
        "Do not invent or preserve unsupported business facts from the invalid output. "
        "If the invalid output contains both request_sections and answer decisions, return only request_sections.\n"
        f"Allowed sections:\n{section_lines}\n"
        f"Allowed image IDs (optional for answer mode, max 3): {', '.join(image_ids)}"
    )
    repair_prompt = (
        "The previous output was invalid. Reformat it to valid JSON only.\n\n"
        f"Previous output:\n{raw}"
    )
    logger.info(
        "LLM selector repair prompt system_prompt_chars=%s history_turns=%s user_prompt_chars=%s",
        chars(repair_system),
        len(history or []),
        chars(repair_prompt),
    )
    repaired_raw = await provider.generate_reply(
        prompt=repair_prompt,
        system_prompt=repair_system,
        history=history or [],
        response_schema=_decision_response_schema(allow_section_requests),
    )
    repaired_decision = _select_decision(
        repaired_raw,
        allowed,
        allowed_images,
        allow_section_requests,
    )
    if repaired_decision:
        return repaired_decision

    return {"mode": "answer", "answer": "Sorry, I could not process that request right now."}


decide_answer_or_sections = decide_next_action
