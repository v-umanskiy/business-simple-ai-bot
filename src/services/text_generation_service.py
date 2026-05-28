import logging
from typing import Any

from src.ai.model_provider import get_model_provider

logger = logging.getLogger(__name__)


def _preview(value: Any, *, max_length: int) -> str:
    text = str(value or "")
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


def _history_preview(history: list[dict] | None) -> list[dict]:
    preview: list[dict] = []
    for turn in history or []:
        role = turn.get("role")
        text = turn.get("text")
        preview.append(
            {
                "role": role if isinstance(role, str) else "",
                "text": _preview(text, max_length=120),
            }
        )
    return preview


async def generate_reply(
    text: str,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> str:
    logger.info(
        "LLM prompt system_prompt=%s history_turns=%s history_preview=%s user_prompt=%s",
        _preview(system_prompt, max_length=8000),
        len(history or []),
        _history_preview(history),
        _preview(text, max_length=4000),
    )
    provider = get_model_provider()
    return await provider.generate_reply(text, system_prompt, history)
