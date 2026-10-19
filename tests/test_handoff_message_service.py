import unittest

from src.services.handoff_message_service import resolve_handoff_message
from src.utils.constants import DEMO_HANDOFF_MESSAGES, REAL_HANDOFF_MESSAGES


class ResolveHandoffMessageTests(unittest.TestCase):
    CONFIG = {"handoff_message": {"de": "Wir melden uns.", "en": "We will reply."}}

    def test_configured_message_is_used_for_the_requested_language(self):
        self.assertEqual(resolve_handoff_message(self.CONFIG, "de"), "Wir melden uns.")

    def test_language_is_normalized_to_its_primary_subtag(self):
        self.assertEqual(resolve_handoff_message(self.CONFIG, " DE-at "), "Wir melden uns.")

    def test_configured_keys_are_trimmed_and_lowercased(self):
        config = {"handoff_message": {" DE ": "Wir melden uns."}}

        self.assertEqual(resolve_handoff_message(config, "de"), "Wir melden uns.")

    def test_unconfigured_language_falls_back_to_configured_english(self):
        self.assertEqual(resolve_handoff_message(self.CONFIG, "fr"), "We will reply.")

    def test_missing_language_means_english(self):
        for lang in (None, "", "   "):
            with self.subTest(lang=lang):
                self.assertEqual(resolve_handoff_message(self.CONFIG, lang), "We will reply.")

    def test_blank_configured_messages_fall_through_to_defaults(self):
        config = {"handoff_message": {"de": "   ", "en": ""}}

        self.assertEqual(resolve_handoff_message(config, "de"), REAL_HANDOFF_MESSAGES["de"])

    def test_non_string_entries_and_non_dict_config_are_ignored(self):
        self.assertEqual(
            resolve_handoff_message({"handoff_message": {"de": 5, 7: "x"}}, "de"),
            REAL_HANDOFF_MESSAGES["de"],
        )
        self.assertEqual(
            resolve_handoff_message({"handoff_message": "text"}, "en"),
            REAL_HANDOFF_MESSAGES["en"],
        )

    def test_real_business_without_config_gets_the_real_defaults(self):
        self.assertEqual(resolve_handoff_message({}, "de"), REAL_HANDOFF_MESSAGES["de"])
        self.assertEqual(resolve_handoff_message({}, "en"), REAL_HANDOFF_MESSAGES["en"])

    def test_demo_business_gets_the_demo_defaults(self):
        config = {"demo": True}

        self.assertEqual(resolve_handoff_message(config, "de"), DEMO_HANDOFF_MESSAGES["de"])
        self.assertEqual(resolve_handoff_message(config, "en"), DEMO_HANDOFF_MESSAGES["en"])

    def test_language_without_a_default_uses_english_default(self):
        self.assertEqual(resolve_handoff_message({}, "ru"), REAL_HANDOFF_MESSAGES["en"])

    def test_configured_message_beats_the_demo_default(self):
        config = {**self.CONFIG, "demo": True}

        self.assertEqual(resolve_handoff_message(config, "en"), "We will reply.")


if __name__ == "__main__":
    unittest.main()
