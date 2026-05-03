from src.ai.model_provider import get_model_provider


async def generate_reply(
    text: str,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> str:
    provider = get_model_provider()
    return await provider.generate_reply(text, system_prompt, history)
