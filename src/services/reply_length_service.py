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


def validate_reply_length(
    *,
    business_id: str,
    user_id: str,
    reply_text: str,
    max_reply_chars: int,
) -> bool:
    is_valid = len(reply_text) <= max_reply_chars
    if not is_valid:
        logger.warning(
            "AI answer exceeded channel limit business_id=%s user_id=%s answer_chars=%s max_reply_chars=%s",
            business_id,
            user_id,
            len(reply_text),
            max_reply_chars,
        )
    return is_valid


async def shorten_reply(
    *,
    message_text: str,
    reply_text: str,
    max_reply_chars: int,
) -> str:
    if len(reply_text) <= max_reply_chars:
        return reply_text

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
    return normalize_plain_text_reply(shortened)


async def repair_reply_length(
    *,
    business_id: str,
    user_id: str,
    retry_count: int,
    message_text: str,
    reply_text: str,
    max_reply_chars: int,
) -> str:
    logger.info(
        "AI answer length repair started business_id=%s user_id=%s retry_count=%s answer_chars=%s max_reply_chars=%s",
        business_id,
        user_id,
        retry_count,
        len(reply_text),
        max_reply_chars,
    )
    shortened = await shorten_reply(
        message_text=message_text,
        reply_text=reply_text,
        max_reply_chars=max_reply_chars,
    )
    logger.info(
        "AI answer length repair finished business_id=%s user_id=%s retry_count=%s answer_chars=%s max_reply_chars=%s",
        business_id,
        user_id,
        retry_count,
        len(shortened),
        max_reply_chars,
    )
    return shortened


def truncate_reply_after_length_repair(
    *,
    business_id: str,
    user_id: str,
    reply_text: str,
    max_reply_chars: int,
) -> str:
    logger.warning(
        "AI answer truncated after length repair business_id=%s user_id=%s answer_chars=%s max_reply_chars=%s",
        business_id,
        user_id,
        len(reply_text),
        max_reply_chars,
    )
    return truncate_reply(reply_text, max_reply_chars)
