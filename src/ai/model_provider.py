import os
from abc import ABC, abstractmethod


class ModelProvider(ABC):
    @abstractmethod
    async def generate_reply(self, prompt: str) -> str:
        raise NotImplementedError


_provider: ModelProvider | None = None


def get_model_provider() -> ModelProvider:
    global _provider
    if _provider is None:
        _provider = _build_provider()
    return _provider


def _build_provider() -> ModelProvider:
    provider_name = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider_name == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider()

    from .openai_provider import OpenAIProvider

    return OpenAIProvider()
