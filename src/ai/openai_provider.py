import asyncio
import os

from openai import OpenAI

from src.ai.model_provider import ModelProvider
from src.services.secret_service import get_secret


class OpenAIProvider(ModelProvider):
    def __init__(self):
        self._model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self._client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    async def generate_reply(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> str:
        def _call() -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                for turn in history:
                    role = turn.get("role")
                    text = turn.get("text")
                    if role in {"user", "assistant"} and text:
                        messages.append({"role": role, "content": text})
            messages.append({"role": "user", "content": prompt})
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
            return (response.choices[0].message.content or "").strip() or "No response"

        return await asyncio.to_thread(_call)
