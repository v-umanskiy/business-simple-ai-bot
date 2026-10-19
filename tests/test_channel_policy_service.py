import unittest

from src.services.channel_policy_service import resolve_channel_policy


class ResolveChannelPolicyTests(unittest.TestCase):
    def test_known_channels_get_their_reply_limit(self):
        for channel, limit in (("widget", 1000), ("telegram", 1000), ("whatsapp", 500)):
            with self.subTest(channel=channel):
                policy = resolve_channel_policy({"channel": channel})

                self.assertEqual(policy.channel, channel)
                self.assertEqual(policy.max_reply_chars, limit)
                self.assertFalse(policy.used_default)

    def test_channel_is_trimmed_and_case_insensitive(self):
        policy = resolve_channel_policy({"channel": "  WhatsApp "})

        self.assertEqual(policy.channel, "whatsapp")
        self.assertEqual(policy.max_reply_chars, 500)

    def test_platform_aliases_are_accepted(self):
        for key in ("platform", "plattform"):
            with self.subTest(key=key):
                policy = resolve_channel_policy({key: "telegram"})

                self.assertEqual(policy.channel, "telegram")
                self.assertFalse(policy.used_default)

    def test_channel_key_wins_over_platform_aliases(self):
        policy = resolve_channel_policy({"channel": "telegram", "platform": "whatsapp"})

        self.assertEqual(policy.channel, "telegram")

    def test_unknown_channel_falls_back_to_widget_and_is_flagged(self):
        policy = resolve_channel_policy({"channel": "sms"})

        self.assertEqual(policy.channel, "widget")
        self.assertEqual(policy.max_reply_chars, 1000)
        self.assertEqual(policy.raw_channel, "sms")
        self.assertTrue(policy.used_default)

    def test_missing_or_empty_channel_falls_back_without_a_flag(self):
        for payload in ({}, {"channel": None}, {"channel": ""}):
            with self.subTest(payload=payload):
                policy = resolve_channel_policy(payload)

                self.assertEqual(policy.channel, "widget")
                self.assertFalse(policy.used_default)


if __name__ == "__main__":
    unittest.main()
