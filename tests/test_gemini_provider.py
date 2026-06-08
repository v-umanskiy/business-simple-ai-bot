import asyncio
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

fake_google = ModuleType("google")
fake_genai = ModuleType("google.genai")
fake_genai.Client = MagicMock()
fake_google.genai = fake_genai
sys.modules.setdefault("google", fake_google)
sys.modules.setdefault("google.genai", fake_genai)

from src.ai.gemini_provider import GeminiProvider


class GeminiProviderTests(unittest.TestCase):
    def _provider_with_client(self):
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text='{"mode":"answer"}')
        with patch("src.ai.gemini_provider.genai.Client", return_value=client), patch(
            "src.ai.gemini_provider.get_secret",
            return_value="test-key",
        ):
            provider = GeminiProvider()
        return provider, client

    def test_generate_reply_uses_system_instruction_and_structured_output(self):
        provider, client = self._provider_with_client()
        schema = {
            "type": "object",
            "properties": {"mode": {"type": "string"}},
            "required": ["mode"],
            "additionalProperties": False,
        }

        result = asyncio.run(
            provider.generate_reply(
                "question",
                system_prompt="rules",
                history=[{"role": "assistant", "text": "previous answer"}],
                response_schema=schema,
            )
        )

        self.assertEqual(result, '{"mode":"answer"}')
        call = client.models.generate_content.call_args.kwargs
        self.assertEqual(call["config"]["system_instruction"], "rules")
        self.assertEqual(call["config"]["response_mime_type"], "application/json")
        self.assertEqual(call["config"]["response_json_schema"], schema)
        self.assertEqual(
            call["contents"],
            [
                {"role": "model", "parts": [{"text": "previous answer"}]},
                {"role": "user", "parts": [{"text": "question"}]},
            ],
        )

    def test_generate_reply_omits_config_without_system_prompt_or_schema(self):
        provider, client = self._provider_with_client()

        asyncio.run(provider.generate_reply("question"))

        self.assertNotIn("config", client.models.generate_content.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
