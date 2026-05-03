import asyncio
import os

from google import genai

from src.ai.model_provider import ModelProvider
from src.services.secret_service import get_secret


class GeminiProvider(ModelProvider):
    def __init__(self):
        self._model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))

    async def generate_reply(self, prompt: str, system_prompt: str | None = None) -> str:
        def _call() -> str:
            contents = []
            if system_prompt:
                contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "user", "parts": [{"text": prompt}]})
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
            )
            return (response.text or "").strip() or "No response"

        return await asyncio.to_thread(_call)
