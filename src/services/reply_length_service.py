import logging

from src.services.reply_format_service import normalize_plain_text_reply
from src.services.text_generation_service import generate_reply

logger = logging.getLogger(__name__)


def add_reply_limit_instruction(system_prompt: str, max_reply_chars: int) -> str:
    return (
        f"{system_prompt}\n\n"
        f"Your final answer must be no longer than {max_reply_chars} characters."
    )


def truncate_reply(reply_text: str, max_reply_chars: int) -> str:
    if len(reply_text) <= max_reply_chars:
        return reply_text
    return f"{reply_text[: max_reply_chars - 3]}..."


async def ensure_reply_length(
    *,
    business_id: str,
    user_id: str,
    message_text: str,
    reply_text: str,
    max_reply_chars: int,
) -> str:
    if len(reply_text) <= max_reply_chars:
        return reply_text

    logger.warning(
        "AI answer exceeded channel limit business_id=%s user_id=%s answer_chars=%s max_reply_chars=%s",
        business_id,
        user_id,
        len(reply_text),
        max_reply_chars,
    )

    shortened = await generate_reply(
        (
            "Rewrite the assistant answer so it is complete, natural, and no longer than "
            f"{max_reply_chars} characters. Return only the shortened answer.\n\n"
            f"User message:\n{message_text}\n\n"
            f"Assistant answer:\n{reply_text}"
        ),
        (
            f"Your answer must be no longer than {max_reply_chars} characters. "
            "Do not mention this character limit."
        ),
        [],
    )
    return truncate_reply(normalize_plain_text_reply(shortened), max_reply_chars)
