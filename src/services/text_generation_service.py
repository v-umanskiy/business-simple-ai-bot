import logging

from src.ai.model_provider import get_model_provider
from src.utils.log_utils import chars

logger = logging.getLogger(__name__)


async def generate_reply(
    text: str,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> str:
    logger.info(
        "LLM prompt system_prompt_chars=%s history_turns=%s user_prompt_chars=%s",
        chars(system_prompt),
        len(history or []),
        chars(text),
    )
    provider = get_model_provider()
    return await provider.generate_reply(text, system_prompt, history)
