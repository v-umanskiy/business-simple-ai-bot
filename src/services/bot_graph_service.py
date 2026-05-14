import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from src.services.business_config_service import get_business_config
from src.services.business_image_catalog_service import get_image_url_map
from src.services.business_knowledge_sections_service import (
    filter_sections,
    get_section_names,
    get_sections_map_by_names,
)
from src.services.handoff_message_service import resolve_handoff_message
from src.services.history_service import load_model_history, prune_history, save_turn
from src.services.knowledge_selection_service import decide_answer_or_sections
from src.services.text_generation_service import generate_reply

logger = logging.getLogger(__name__)


class BotState(TypedDict, total=False):
    business_id: str
    user_id: str
    message_text: str
    payload: dict[str, Any]
    chat_key: str
    business_config: dict[str, Any]
    base_system_prompt: str
    system_prompt: str
    section_names: list[str]
    image_ids: list[str]
    image_url_map: dict[str, str]
    image_urls: list[str]
    history: list[dict]
    branch: str
    lang: str
    reply_text: str
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

    section_names = get_section_names(business_id)
    image_url_map = get_image_url_map(business_id)
    image_ids = list(image_url_map.keys())

    chat_key = f"{business_id}#{user_id}"
    history = await load_model_history(chat_key)

    state["business_config"] = business_config
    state["base_system_prompt"] = system_prompt
    state["system_prompt"] = system_prompt
    state["section_names"] = section_names
    state["image_ids"] = image_ids
    state["image_url_map"] = image_url_map
    state["image_urls"] = []
    state["chat_key"] = chat_key
    state["history"] = history
    logger.info(
        "Bot context loaded business_id=%s user_id=%s knowledge_sections=%s image_ids=%s history_turns=%s",
        business_id,
        user_id,
        section_names,
        image_ids,
        len(history),
    )
    return state


async def _generate(state: BotState) -> BotState:
    if state.get("error"):
        return state

    message_text = state["message_text"]
    base_system_prompt = state["base_system_prompt"]
    section_names = state.get("section_names") or []
    image_ids = state.get("image_ids") or []
    history = state.get("history") or []

    decision = await decide_answer_or_sections(
        message_text=message_text,
        base_system_prompt=base_system_prompt,
        section_names=section_names,
        image_ids=image_ids,
        history=history,
    )

    mode = str(decision.get("mode") or "")
    if mode == "answer":
        state["branch"] = "reply_text"
        state["reply_text"] = str(decision.get("answer") or "")
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
    if mode == "handoff":
        state["branch"] = "handoff"
        state["lang"] = str(decision.get("lang") or "en")
        state["reply_text"] = resolve_handoff_message(state.get("business_config") or {}, state["lang"])
        logger.info(
            "AI decision business_id=%s user_id=%s mode=handoff lang=%s answer=%s",
            state["business_id"],
            state["user_id"],
            state["lang"],
            _preview(state["reply_text"]),
        )
        return state

    selected_names = [str(name) for name in (decision.get("sections") or [])]
    logger.info(
        "AI decision business_id=%s user_id=%s mode=request_sections knowledge_sections=%s",
        state["business_id"],
        state["user_id"],
        selected_names,
    )
    sections_map = get_sections_map_by_names(state["business_id"], selected_names)
    selected_sections = filter_sections(sections_map, selected_names)
    blocks = [f"[{name}]\n{text}" for name, text in selected_sections.items()]
    sections_text = "\n\n".join(blocks)
    effective_system_prompt = (
        f"{base_system_prompt}\n\n"
        "Use the provided business knowledge sections as the primary source for factual answers. "
        "If required information is missing in these sections, say that clearly.\n\n"
        f"Provided knowledge sections:\n{sections_text}"
    )

    state["reply_text"] = await generate_reply(
        message_text,
        effective_system_prompt,
        state.get("history") or [],
    )
    state["branch"] = "reply_text"
    state["lang"] = ""
    logger.info(
        "AI answer business_id=%s user_id=%s knowledge_sections=%s answer=%s",
        state["business_id"],
        state["user_id"],
        list(selected_sections.keys()),
        _preview(state["reply_text"]),
    )
    return state


async def _persist(state: BotState) -> BotState:
    if state.get("error"):
        return state

    chat_key = state["chat_key"]
    message_text = state["message_text"]
    reply_text = state["reply_text"]

    await save_turn(chat_key, "user", message_text)
    await save_turn(chat_key, "assistant", reply_text)
    await prune_history(chat_key)
    return state


_graph = None


def _build_graph():
    graph = StateGraph(BotState)
    graph.add_node("load_context", _load_context)
    graph.add_node("generate", _generate)
    graph.add_node("persist", _persist)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "generate")
    graph.add_edge("generate", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


async def run_bot_flow(payload: dict[str, Any]) -> dict[str, Any]:
    state: BotState = {
        "business_id": str(payload.get("business_id") or ""),
        "user_id": str(payload.get("user_id") or ""),
        "message_text": str(payload.get("message_text") or ""),
        "payload": payload,
    }

    logger.info(
        "Received message business_id=%s user_id=%s message=%s",
        state["business_id"],
        state["user_id"],
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
        "lang": result.get("lang") or "",
        "reply_text": result.get("reply_text") or "",
        "number_images": number_images,
        "image_url_1": image_url_1,
        "image_url_2": image_url_2,
        "image_url_3": image_url_3,
        "echo": payload,
    }
