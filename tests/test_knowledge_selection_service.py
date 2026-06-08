import unittest

from src.services.knowledge_selection_service import (
    _decision_response_schema,
    _select_decision,
    parse_mode_response,
)


class KnowledgeSelectionServiceTests(unittest.TestCase):
    def test_parse_mode_response_accepts_single_json_object(self):
        self.assertEqual(
            parse_mode_response('{"mode":"request_sections","sections":[3]}'),
            {"mode": "request_sections", "sections": [3]},
        )

    def test_select_decision_prefers_section_request_over_concatenated_answer(self):
        raw = """
        {"mode":"request_sections","sections":[3]}
        {"mode":"answer","answer":"Invented prices","image_ids":[]}
        """

        self.assertEqual(
            _select_decision(raw, {1, 2, 3}, set(), True),
            {"mode": "request_sections", "sections": [3]},
        )

    def test_select_decision_combines_valid_section_requests(self):
        raw = """
        {"mode":"request_sections","sections":[3]}
        {"mode":"request_sections","sections":[3,4,99]}
        """

        self.assertEqual(
            _select_decision(raw, {1, 2, 3, 4}, set(), True),
            {"mode": "request_sections", "sections": [3, 4]},
        )

    def test_select_decision_rejects_ambiguous_multiple_answers(self):
        raw = """
        {"mode":"answer","answer":"First","image_ids":[]}
        {"mode":"answer","answer":"Second","image_ids":[]}
        """

        self.assertEqual(_select_decision(raw, set(), set(), False), {})

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


if __name__ == "__main__":
    unittest.main()
