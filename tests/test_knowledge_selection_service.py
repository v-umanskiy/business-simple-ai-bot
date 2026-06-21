import unittest
from typing import Any

from src.services.knowledge_selection_service import (
    _decision_response_schema,
    _select_decision,
    parse_mode_response,
)


def _contains_title(value: Any) -> bool:
    if isinstance(value, dict):
        return "title" in value or any(_contains_title(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_title(item) for item in value)
    return False


class KnowledgeSelectionServiceTests(unittest.TestCase):
    def test_parse_mode_response_accepts_single_json_object(self):
        self.assertEqual(
            parse_mode_response('{"mode":"request_sections","sections":[3]}'),
            {"mode": "request_sections", "sections": [3]},
        )

    def test_select_decision_prefers_section_request_over_concatenated_answer(self):
        raw = """
        {"mode":"request_sections","answer":null,"image_ids":[],"sections":[3],"lang":null}
        {"mode":"answer","answer":"Invented prices","image_ids":[],"sections":[],"lang":null}
        """

        self.assertEqual(
            _select_decision(raw, {1, 2, 3}, set(), True),
            {"mode": "request_sections", "sections": [3]},
        )

    def test_select_decision_combines_valid_section_requests(self):
        raw = """
        {"mode":"request_sections","answer":null,"image_ids":[],"sections":[3],"lang":null}
        {"mode":"request_sections","answer":null,"image_ids":[],"sections":[3,4,99],"lang":null}
        """

        self.assertEqual(
            _select_decision(raw, {1, 2, 3, 4}, set(), True),
            {"mode": "request_sections", "sections": [3, 4]},
        )

    def test_select_decision_rejects_ambiguous_multiple_answers(self):
        raw = """
        {"mode":"answer","answer":"First","image_ids":[],"sections":[],"lang":null}
        {"mode":"answer","answer":"Second","image_ids":[],"sections":[],"lang":null}
        """

        self.assertEqual(_select_decision(raw, set(), set(), False), {})

    def test_select_decision_rejects_extra_fields(self):
        raw = """
        {"mode":"answer","answer":"Hello","image_ids":[],"sections":[],"lang":null,"extra":"nope"}
        """

        self.assertEqual(_select_decision(raw, set(), set(), False), {})

    def test_select_decision_rejects_invalid_field_types(self):
        raw = """
        {"mode":"answer","answer":["Hello"],"image_ids":[],"sections":[],"lang":null}
        """

        self.assertEqual(_select_decision(raw, set(), set(), False), {})

    def test_select_decision_rejects_section_request_when_not_allowed(self):
        raw = """
        {"mode":"request_sections","answer":null,"image_ids":[],"sections":[3],"lang":null}
        """

        self.assertEqual(_select_decision(raw, {3}, set(), False), {})

    def test_decision_response_schema_allows_section_requests_during_selection(self):
        schema = _decision_response_schema(True)

        self.assertEqual(
            schema["properties"]["mode"]["enum"],
            ["answer", "request_sections", "handoff"],
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_decision_response_schema_excludes_section_requests_from_final_answer(self):
        schema = _decision_response_schema(False)

        self.assertEqual(schema["properties"]["mode"]["enum"], ["answer", "handoff"])

    def test_decision_response_schema_omits_pydantic_titles(self):
        schema = _decision_response_schema(True)

        self.assertFalse(_contains_title(schema))


if __name__ == "__main__":
    unittest.main()
