import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from src.services.business_config_service import get_business_config
from src.services.business_image_catalog_service import get_image_url_map
from src.services.business_knowledge_sections_service import (
    filter_sections,
    get_section_options,
    get_sections_map_by_ids,
    map_section_ids_by_numbers,
)
from src.services.channel_policy_service import resolve_channel_policy
from src.services.handoff_message_service import resolve_handoff_message
from src.services.history_service import load_model_history, prune_history, save_turn
from src.services.knowledge_selection_service import decide_next_action
from src.services.reply_format_service import normalize_plain_text_reply
from src.services.reply_length_service import (
    add_reply_limit_instruction,
    repair_reply_length,
    truncate_reply,
    truncate_reply_after_length_repair,
    validate_reply_length,
)

logger = logging.getLogger(__name__)


class BotState(TypedDict, total=False):
    business_id: str
    user_id: str
    message_text: str
    channel: str
    max_reply_chars: int
    payload: dict[str, Any]
    chat_key: str
    business_config: dict[str, Any]
    base_system_prompt: str
    system_prompt: str
    section_options: list[dict]
    image_ids: list[str]
    image_url_map: dict[str, str]
    image_urls: list[str]
    history: list[dict]
    branch: str
    lang: str
    reply_text: str
    length_retry_count: int
    length_valid: bool
    error: str


def _preview(value: Any, *, max_length: int = 1200) -> str:
    text = str(value or "")
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


async def _load_context(state: BotState) -> BotState:
    business_id = state["business_id"]
    user_id = state["user_id"]
    business_config = get_business_config(business_id)
    if not business_config:
        logger.warning(
            "Bot context load failed business_id=%s user_id=%s reason=missing_business_config",
            business_id,
            user_id,
        )
        state["error"] = "Business configuration was not found."
        return state
    system_prompt = business_config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        logger.warning(
            "Bot context load failed business_id=%s user_id=%s reason=missing_system_prompt",
            business_id,
            user_id,
        )
        state["error"] = "Business configuration was not found."
        return state

    section_options = get_section_options(business_id)
    image_url_map = get_image_url_map(business_id)
    image_ids = list(image_url_map.keys())

    chat_key = f"{business_id}#{user_id}"
    history = await load_model_history(chat_key)

    state["business_config"] = business_config
    state["base_system_prompt"] = add_reply_limit_instruction(system_prompt, state["max_reply_chars"])
    state["system_prompt"] = state["base_system_prompt"]
    state["section_options"] = section_options
    state["image_ids"] = image_ids
    state["image_url_map"] = image_url_map
    state["image_urls"] = []
    state["chat_key"] = chat_key
    state["history"] = history
    logger.info(
        "Bot context loaded business_id=%s user_id=%s channel=%s max_reply_chars=%s knowledge_sections=%s image_ids=%s history_turns=%s",
        business_id,
        user_id,
        state["channel"],
        state["max_reply_chars"],
        [option.get("title") for option in section_options],
        image_ids,
        len(history),
    )
    return state


async def _generate(state: BotState) -> BotState:
    if state.get("error"):
        return state

    message_text = state["message_text"]
    base_system_prompt = state["base_system_prompt"]
    section_options = state.get("section_options") or []
    image_ids = state.get("image_ids") or []
    history = state.get("history") or []
    loaded_section_ids: list[str] = []
    loaded_sections: list[dict] = []

    for section_request_count in range(3):
        remaining_options = [
            option for option in section_options if option["section_id"] not in loaded_section_ids
        ]
        if not remaining_options:
            break

        decision = await decide_next_action(
            message_text=message_text,
            base_system_prompt=base_system_prompt,
            section_options=remaining_options,
            image_ids=image_ids,
            history=history,
            provided_sections=loaded_sections,
            allow_section_requests=True,
        )

        mode = str(decision.get("mode") or "")
        if mode == "answer":
            return _apply_answer_decision(state, decision)
        if mode == "handoff":
            return _apply_handoff_decision(state, decision)

        selected_numbers = [number for number in (decision.get("sections") or []) if isinstance(number, int)]
        selected_ids = map_section_ids_by_numbers(remaining_options, selected_numbers)
        logger.info(
            "AI decision business_id=%s user_id=%s mode=request_sections round=%s knowledge_sections=%s",
            state["business_id"],
            state["user_id"],
            section_request_count + 1,
            selected_numbers,
        )
        if not selected_ids:
            break

        sections_map = get_sections_map_by_ids(state["business_id"], selected_ids)
        selected_sections = filter_sections(sections_map, selected_ids)
        loaded_section_ids.extend(selected_ids)
        loaded_sections.extend(selected_sections)

    final_decision = await decide_next_action(
        message_text=message_text,
        base_system_prompt=base_system_prompt,
        section_options=[],
        image_ids=image_ids,
        history=history,
        provided_sections=loaded_sections,
        allow_section_requests=False,
    )
    if final_decision.get("mode") == "handoff":
        return _apply_handoff_decision(state, final_decision)
    return _apply_answer_decision(state, final_decision)


def _apply_answer_decision(state: BotState, decision: dict[str, Any]) -> BotState:
    raw_reply_text = str(decision.get("answer") or "")
    logger.info(
        "AI raw answer business_id=%s user_id=%s mode=answer answer=%s",
        state["business_id"],
        state["user_id"],
        _preview(raw_reply_text),
    )
    state["branch"] = "reply_text"
    state["reply_text"] = normalize_plain_text_reply(raw_reply_text)
    state["lang"] = ""
    selected_image_ids = [str(x) for x in (decision.get("image_ids") or [])]
    state["image_ids"] = selected_image_ids
    image_url_map = state.get("image_url_map") or {}
    state["image_urls"] = [image_url_map[image_id] for image_id in selected_image_ids if image_id in image_url_map]
    logger.info(
        "AI decision business_id=%s user_id=%s mode=answer image_ids=%s answer=%s",
        state["business_id"],
        state["user_id"],
        selected_image_ids,
        _preview(state["reply_text"]),
    )
    return state


def _apply_handoff_decision(state: BotState, decision: dict[str, Any]) -> BotState:
    state["branch"] = "handoff"
    state["lang"] = str(decision.get("lang") or "en")
    raw_reply_text = resolve_handoff_message(state.get("business_config") or {}, state["lang"])
    logger.info(
        "AI raw answer business_id=%s user_id=%s mode=handoff answer=%s",
        state["business_id"],
        state["user_id"],
        _preview(raw_reply_text),
    )
    state["reply_text"] = normalize_plain_text_reply(raw_reply_text)
    state["reply_text"] = truncate_reply(state["reply_text"], state["max_reply_chars"])
    logger.info(
        "AI decision business_id=%s user_id=%s mode=handoff lang=%s answer=%s",
        state["business_id"],
        state["user_id"],
        state["lang"],
        _preview(state["reply_text"]),
    )
    return state


async def _validate_length(state: BotState) -> BotState:
    if state.get("error"):
        return state

    state["length_valid"] = validate_reply_length(
        business_id=state["business_id"],
        user_id=state["user_id"],
        reply_text=state.get("reply_text") or "",
        max_reply_chars=state["max_reply_chars"],
    )
    return state


async def _repair_length(state: BotState) -> BotState:
    if state.get("error"):
        return state

    state["length_retry_count"] = state.get("length_retry_count", 0) + 1
    state["reply_text"] = await repair_reply_length(
        business_id=state["business_id"],
        user_id=state["user_id"],
        retry_count=state["length_retry_count"],
        message_text=state["message_text"],
        reply_text=state.get("reply_text") or "",
        max_reply_chars=state["max_reply_chars"],
    )
    return state


async def _truncate_length(state: BotState) -> BotState:
    if state.get("error"):
        return state

    state["reply_text"] = truncate_reply_after_length_repair(
        business_id=state["business_id"],
        user_id=state["user_id"],
        reply_text=state.get("reply_text") or "",
        max_reply_chars=state["max_reply_chars"],
    )
    state["length_valid"] = True
    return state


def _route_after_length_validation(state: BotState) -> str:
    if state.get("error") or state.get("length_valid"):
        return "persist"
    if state.get("length_retry_count", 0) < 1:
        return "repair_length"
    return "truncate_length"


async def _persist(state: BotState) -> BotState:
    if state.get("error"):
        return state

    chat_key = state["chat_key"]
    message_text = state["message_text"]
    reply_text = state["reply_text"]

    try:
        await save_turn(chat_key, "user", message_text)
        await save_turn(chat_key, "assistant", reply_text)
        await prune_history(chat_key)
    except Exception:
        logger.exception(
            "History persistence failed business_id=%s user_id=%s chat_key=%s",
            state["business_id"],
            state["user_id"],
            chat_key,
        )
    return state


_graph = None


def _build_graph():
    graph = StateGraph(BotState)
    graph.add_node("load_context", _load_context)
    graph.add_node("generate", _generate)
    graph.add_node("validate_length", _validate_length)
    graph.add_node("repair_length", _repair_length)
    graph.add_node("truncate_length", _truncate_length)
    graph.add_node("persist", _persist)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "generate")
    graph.add_edge("generate", "validate_length")
    graph.add_conditional_edges(
        "validate_length",
        _route_after_length_validation,
        {
            "persist": "persist",
            "repair_length": "repair_length",
            "truncate_length": "truncate_length",
        },
    )
    graph.add_edge("repair_length", "validate_length")
    graph.add_edge("truncate_length", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


async def run_bot_flow(payload: dict[str, Any]) -> dict[str, Any]:
    channel_policy = resolve_channel_policy(payload)
    state: BotState = {
        "business_id": str(payload.get("business_id") or ""),
        "user_id": str(payload.get("user_id") or ""),
        "message_text": str(payload.get("message_text") or ""),
        "channel": channel_policy.channel,
        "max_reply_chars": channel_policy.max_reply_chars,
        "payload": payload,
        "length_retry_count": 0,
    }

    if channel_policy.used_default:
        logger.warning(
            "Unknown channel received business_id=%s user_id=%s raw_channel=%s fallback_channel=%s",
            state["business_id"],
            state["user_id"],
            channel_policy.raw_channel,
            channel_policy.channel,
        )

    logger.info(
        "Received message business_id=%s user_id=%s channel=%s max_reply_chars=%s message=%s",
        state["business_id"],
        state["user_id"],
        state["channel"],
        state["max_reply_chars"],
        _preview(state["message_text"]),
    )

    result = await get_graph().ainvoke(state)

    if result.get("error"):
        logger.warning(
            "Bot flow failed business_id=%s user_id=%s error=%s",
            state["business_id"],
            state["user_id"],
            result["error"],
        )
        return {
            "ok": False,
            "branch": "error",
            "channel": state["channel"],
            "max_reply_chars": state["max_reply_chars"],
            "reply_text": result["error"],
            "number_images": 0,
            "image_url_1": "",
            "image_url_2": "",
            "image_url_3": "",
            "echo": payload,
        }

    image_urls = list(result.get("image_urls") or [])
    number_images = len(image_urls)
    image_url_1 = image_urls[0] if number_images >= 1 else ""
    image_url_2 = image_urls[1] if number_images >= 2 else ""
    image_url_3 = image_urls[2] if number_images >= 3 else ""

    return {
        "ok": True,
        "branch": result.get("branch") or "reply_text",
        "channel": result.get("channel") or state["channel"],
        "max_reply_chars": result.get("max_reply_chars") or state["max_reply_chars"],
        "lang": result.get("lang") or "",
        "reply_text": result.get("reply_text") or "",
        "number_images": number_images,
        "image_url_1": image_url_1,
        "image_url_2": image_url_2,
        "image_url_3": image_url_3,
        "echo": payload,
    }
