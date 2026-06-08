import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ai.openai_provider import OpenAIProvider


class OpenAIProviderTests(unittest.TestCase):
    def _provider_with_client(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"mode":"answer"}'))]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = completion
        with patch("src.ai.openai_provider.OpenAI", return_value=client), patch(
            "src.ai.openai_provider.get_secret",
            return_value="test-key",
        ):
            provider = OpenAIProvider()
        return provider, client

    def test_generate_reply_uses_structured_outputs_when_schema_is_provided(self):
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
                response_schema=schema,
            )
        )

        self.assertEqual(result, '{"mode":"answer"}')
        response_format = client.chat.completions.create.call_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], schema)

    def test_generate_reply_omits_response_format_without_schema(self):
        provider, client = self._provider_with_client()

        asyncio.run(provider.generate_reply("question", system_prompt="rules"))

        self.assertNotIn("response_format", client.chat.completions.create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
