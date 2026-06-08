import asyncio
import os

from google import genai

from src.ai.model_provider import ModelProvider
from src.services.secret_service import get_secret


class GeminiProvider(ModelProvider):
    def __init__(self):
        self._model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))

    async def generate_reply(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> str:
        def _call() -> str:
            contents = []
            if history:
                for turn in history:
                    role = turn.get("role")
                    text = turn.get("text")
                    if not text:
                        continue
                    gemini_role = "model" if role == "assistant" else "user"
                    contents.append({"role": gemini_role, "parts": [{"text": text}]})
            contents.append({"role": "user", "parts": [{"text": prompt}]})
            config = {}
            if system_prompt:
                config["system_instruction"] = system_prompt
            if response_schema:
                config["response_mime_type"] = "application/json"
                config["response_json_schema"] = response_schema
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                **({"config": config} if config else {}),
            )
            return (response.text or "").strip() or "No response"

        return await asyncio.to_thread(_call)
