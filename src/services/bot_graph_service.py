from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from src.services.business_config_service import get_system_prompt
from src.services.history_service import load_model_history, prune_history, save_turn
from src.services.text_generation_service import generate_reply


class BotState(TypedDict, total=False):
    business_id: str
    user_id: str
    message_text: str
    payload: dict[str, Any]
    chat_key: str
    system_prompt: str
    history: list[dict]
    reply_text: str
    error: str


async def _load_context(state: BotState) -> BotState:
    business_id = state["business_id"]
    user_id = state["user_id"]

    system_prompt = get_system_prompt(business_id)
    if not system_prompt:
        state["error"] = "Business configuration was not found."
        return state

    chat_key = f"{business_id}#{user_id}"
    history = await load_model_history(chat_key)

    state["system_prompt"] = system_prompt
    state["chat_key"] = chat_key
    state["history"] = history
    return state


async def _generate(state: BotState) -> BotState:
    if state.get("error"):
        return state

    reply_text = await generate_reply(
        state["message_text"],
        state["system_prompt"],
        state.get("history") or [],
    )
    state["reply_text"] = reply_text
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

    result = await get_graph().ainvoke(state)

    if result.get("error"):
        return {
            "ok": False,
            "branch": "error",
            "reply_text": result["error"],
            "echo": payload,
        }

    return {
        "ok": True,
        "branch": "reply_text",
        "reply_text": result.get("reply_text") or "",
        "echo": payload,
    }
