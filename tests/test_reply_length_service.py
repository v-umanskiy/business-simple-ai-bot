import unittest
from unittest.mock import AsyncMock, patch

from src.services.reply_format_service import normalize_plain_text_reply
from src.services.reply_length_service import (
    add_reply_limit_instruction,
    repair_reply_length,
    shorten_reply,
    truncate_reply,
    truncate_reply_after_length_repair,
    validate_reply_length,
)

GENERATE = "src.services.reply_length_service.generate_reply"


class NormalizePlainTextReplyTests(unittest.TestCase):
    def test_markdown_bold_markers_are_removed_and_text_is_trimmed(self):
        self.assertEqual(normalize_plain_text_reply("  **Hello** *world*  \n"), "Hello *world*")

    def test_empty_values_become_an_empty_string(self):
        self.assertEqual(normalize_plain_text_reply(None), "")
        self.assertEqual(normalize_plain_text_reply(""), "")


class LimitInstructionTests(unittest.TestCase):
    def test_instruction_is_appended_to_the_system_prompt(self):
        result = add_reply_limit_instruction("Be nice.", 500)

        self.assertTrue(result.startswith("Be nice.\n\n"))
        self.assertTrue(result.endswith("no longer than 500 characters."))


class TruncateReplyTests(unittest.TestCase):
    def test_reply_within_the_limit_is_unchanged(self):
        self.assertEqual(truncate_reply("x" * 10, 10), "x" * 10)

    def test_longer_reply_is_cut_to_exactly_the_limit_with_an_ellipsis(self):
        result = truncate_reply("x" * 11, 10)

        self.assertEqual(result, "x" * 7 + "...")
        self.assertEqual(len(result), 10)

    def test_truncation_after_repair_uses_the_same_rule(self):
        result = truncate_reply_after_length_repair(
            business_id="b", user_id="u", reply_text="x" * 50, max_reply_chars=20
        )

        self.assertEqual(result, "x" * 17 + "...")


class ValidateReplyLengthTests(unittest.TestCase):
    def test_reply_at_the_limit_is_valid(self):
        self.assertTrue(
            validate_reply_length(business_id="b", user_id="u", reply_text="x" * 10, max_reply_chars=10)
        )

    def test_reply_over_the_limit_is_invalid_and_logged(self):
        with self.assertLogs("src.services.reply_length_service", level="WARNING") as logs:
            valid = validate_reply_length(business_id="b", user_id="u", reply_text="x" * 11, max_reply_chars=10)

        self.assertFalse(valid)
        self.assertIn("exceeded channel limit", logs.output[0])


class ShortenReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_within_the_limit_does_not_call_the_model(self):
        with patch(GENERATE, new_callable=AsyncMock) as generate:
            result = await shorten_reply(message_text="Hi", reply_text="short", max_reply_chars=10)

        self.assertEqual(result, "short")
        generate.assert_not_called()

    async def test_overlong_reply_is_rewritten_by_the_model_and_normalized(self):
        with patch(GENERATE, new_callable=AsyncMock, return_value="  **Shorter** answer ") as generate:
            result = await shorten_reply(message_text="What is it?", reply_text="x" * 50, max_reply_chars=20)

        self.assertEqual(result, "Shorter answer")
        prompt, system_prompt, history = generate.await_args.args
        self.assertIn("no longer than 20 characters", prompt)
        self.assertIn("What is it?", prompt)
        self.assertIn("x" * 50, prompt)
        self.assertIn("no longer than 20 characters", system_prompt)
        self.assertEqual(history, [])

    async def test_repair_returns_the_shortened_reply(self):
        with patch(GENERATE, new_callable=AsyncMock, return_value="Short."):
            result = await repair_reply_length(
                business_id="b",
                user_id="u",
                retry_count=1,
                message_text="Hi",
                reply_text="x" * 50,
                max_reply_chars=20,
            )

        self.assertEqual(result, "Short.")


if __name__ == "__main__":
    unittest.main()
