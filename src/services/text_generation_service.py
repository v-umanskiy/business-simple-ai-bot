from src.ai.model_provider import get_model_provider


async def generate_reply(text: str) -> str:
    provider = get_model_provider()
    return await provider.generate_reply(text)
