import unittest
from unittest.mock import AsyncMock, patch

from src.services.text_generation_service import generate_reply
from src.utils.log_utils import chars


class CharsTests(unittest.TestCase):
    def test_length_of_text(self):
        self.assertEqual(chars("My phone is 12345"), 17)

    def test_empty_values_have_no_characters(self):
        self.assertEqual(chars(None), 0)
        self.assertEqual(chars(""), 0)


class GenerateReplyLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompts_do_not_reach_the_log(self):
        provider = AsyncMock()
        provider.generate_reply.return_value = "ok"
        history = [{"role": "user", "text": "earlier"}]
        with (
            patch("src.services.text_generation_service.get_model_provider", return_value=provider),
            self.assertLogs("src.services.text_generation_service", level="INFO") as logs,
        ):
            reply = await generate_reply("my secret question", "internal system prompt", history)

        self.assertEqual(reply, "ok")
        provider.generate_reply.assert_awaited_once_with("my secret question", "internal system prompt", history)
        logged = "\n".join(logs.output)
        self.assertIn("system_prompt_chars=22", logged)
        self.assertIn("user_prompt_chars=18", logged)
        for text in ("my secret question", "internal system prompt", "earlier"):
            self.assertNotIn(text, logged)


if __name__ == "__main__":
    unittest.main()
