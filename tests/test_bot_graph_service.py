import copy
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, call, patch

from src.services.bot_graph_service import run_bot_flow

MODULE = "src.services.bot_graph_service"

_UNSET = object()

BASE_CONFIG = {"system_prompt": "You are a helpful assistant."}

DEFAULT_PAYLOAD = {"business_id": "biz-1", "user_id": "user-1", "message_text": "Hi"}

SECTION_OPTIONS = [
    {"number": 1, "section_id": "s-prices", "title": "Prices"},
    {"number": 2, "section_id": "s-hours", "title": "Hours"},
    {"number": 3, "section_id": "s-faq", "title": "FAQ"},
    {"number": 4, "section_id": "s-legal", "title": "Legal"},
]

SECTIONS = {
    option["section_id"]: {"title": option["title"], "text": f"{option['title']} text."}
    for option in SECTION_OPTIONS
}

IMAGE_URL_MAP = {"a": "https://img.test/a.png", "b": "https://img.test/b.png"}


def _answer(text="Hello there", image_ids=()):
    return {"mode": "answer", "answer": text, "image_ids": list(image_ids)}


def _request_sections(*numbers):
    return {"mode": "request_sections", "sections": list(numbers)}


def _handoff(lang="en"):
    return {"mode": "handoff", "lang": lang}


class _Decider:
    """Stands in for decide_next_action: replays decisions, records call arguments.

    The graph mutates the provided_sections list between rounds, so arguments are
    deep-copied at call time.
    """

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self._decisions[len(self.calls) - 1]


class _Run:
    def __init__(self, result, decider, save_turn, prune_history, repair):
        self.result = result
        self.decider = decider
        self.save_turn = save_turn
        self.prune_history = prune_history
        self.repair = repair


async def _run_flow(
    payload=None,
    *,
    decisions,
    config=_UNSET,
    section_options=_UNSET,
    history=None,
    repaired=(),
    save_turn_error=None,
):
    payload = {**DEFAULT_PAYLOAD, **(payload or {})}
    decider = _Decider(decisions)
    save_turn = AsyncMock(side_effect=save_turn_error)
    prune_history = AsyncMock()
    repair = AsyncMock(side_effect=list(repaired))

    with ExitStack() as stack:
        stack.enter_context(
            patch(f"{MODULE}.get_business_config", return_value=BASE_CONFIG if config is _UNSET else config)
        )
        stack.enter_context(
            patch(
                f"{MODULE}.get_section_options",
                return_value=SECTION_OPTIONS if section_options is _UNSET else section_options,
            )
        )
        stack.enter_context(patch(f"{MODULE}.get_image_url_map", return_value=IMAGE_URL_MAP))
        stack.enter_context(
            patch(
                f"{MODULE}.get_sections_map_by_ids",
                side_effect=lambda business_id, ids: {i: SECTIONS[i] for i in ids},
            )
        )
        stack.enter_context(patch(f"{MODULE}.load_model_history", AsyncMock(return_value=history or [])))
        stack.enter_context(patch(f"{MODULE}.decide_next_action", decider))
        stack.enter_context(patch(f"{MODULE}.save_turn", save_turn))
        stack.enter_context(patch(f"{MODULE}.prune_history", prune_history))
        stack.enter_context(patch(f"{MODULE}.repair_reply_length", repair))
        result = await run_bot_flow(payload)

    return _Run(result, decider, save_turn, prune_history, repair)


class AnswerFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_answer_is_normalized_and_persisted(self):
        run = await _run_flow(decisions=[_answer("**Hello** there")])

        self.assertTrue(run.result["ok"])
        self.assertEqual(run.result["branch"], "reply_text")
        self.assertEqual(run.result["reply_text"], "Hello there")
        self.assertEqual(run.result["echo"], {**DEFAULT_PAYLOAD})
        self.assertEqual(len(run.decider.calls), 1)
        self.assertTrue(run.decider.calls[0]["allow_section_requests"])
        self.assertEqual(
            run.save_turn.await_args_list,
            [call("biz-1#user-1", "user", "Hi"), call("biz-1#user-1", "assistant", "Hello there")],
        )
        run.prune_history.assert_awaited_once_with("biz-1#user-1")

    async def test_known_images_are_returned_and_unknown_ones_dropped(self):
        run = await _run_flow(decisions=[_answer(image_ids=["a", "missing"])])

        self.assertEqual(run.result["number_images"], 1)
        self.assertEqual(run.result["image_url_1"], "https://img.test/a.png")
        self.assertEqual(run.result["image_url_2"], "")
        self.assertEqual(run.result["image_url_3"], "")

    async def test_channel_limit_is_added_to_system_prompt_and_response(self):
        run = await _run_flow({"channel": "whatsapp"}, decisions=[_answer()])

        self.assertEqual(run.result["channel"], "whatsapp")
        self.assertEqual(run.result["max_reply_chars"], 500)
        self.assertIn("no longer than 500 characters", run.decider.calls[0]["base_system_prompt"])
        self.assertIn(BASE_CONFIG["system_prompt"], run.decider.calls[0]["base_system_prompt"])


class SectionRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_requested_section_is_provided_to_next_round_and_not_offered_again(self):
        run = await _run_flow(decisions=[_request_sections(1), _answer("It costs 100 EUR.")])

        self.assertEqual(run.result["reply_text"], "It costs 100 EUR.")
        first, second = run.decider.calls
        self.assertEqual([o["title"] for o in first["section_options"]], ["Prices", "Hours", "FAQ", "Legal"])
        self.assertEqual(first["provided_sections"], [])
        self.assertEqual([o["title"] for o in second["section_options"]], ["Hours", "FAQ", "Legal"])
        self.assertEqual(second["provided_sections"], [SECTIONS["s-prices"]])
        self.assertTrue(second["allow_section_requests"])

    async def test_section_requests_stop_after_three_rounds_and_final_call_forbids_more(self):
        run = await _run_flow(
            decisions=[
                _request_sections(1),
                _request_sections(2),
                _request_sections(3),
                _answer("Final."),
            ]
        )

        self.assertEqual(run.result["reply_text"], "Final.")
        self.assertEqual([c["allow_section_requests"] for c in run.decider.calls], [True, True, True, False])
        final = run.decider.calls[-1]
        self.assertEqual(final["section_options"], [])
        self.assertEqual(
            [s["title"] for s in final["provided_sections"]],
            ["Prices", "Hours", "FAQ"],
        )

    async def test_loop_ends_when_every_section_has_been_loaded(self):
        run = await _run_flow(
            decisions=[_request_sections(1), _answer("Done.")],
            section_options=SECTION_OPTIONS[:1],
        )

        self.assertEqual(run.result["reply_text"], "Done.")
        self.assertEqual([c["allow_section_requests"] for c in run.decider.calls], [True, False])
        self.assertEqual(run.decider.calls[1]["section_options"], [])
        self.assertEqual(run.decider.calls[1]["provided_sections"], [SECTIONS["s-prices"]])

    async def test_unknown_section_number_goes_straight_to_final_answer(self):
        run = await _run_flow(decisions=[_request_sections(99), _answer("No section needed.")])

        self.assertEqual(run.result["reply_text"], "No section needed.")
        self.assertEqual([c["allow_section_requests"] for c in run.decider.calls], [True, False])
        self.assertEqual(run.decider.calls[1]["provided_sections"], [])

    async def test_no_sections_configured_skips_the_request_loop(self):
        run = await _run_flow(decisions=[_answer("Plain answer.")], section_options=[])

        self.assertEqual(run.result["reply_text"], "Plain answer.")
        self.assertEqual([c["allow_section_requests"] for c in run.decider.calls], [False])


class HandoffTests(unittest.IsolatedAsyncioTestCase):
    CONFIG = {
        **BASE_CONFIG,
        "handoff_message": {"de": "Wir melden uns.", "en": "We will reply."},
    }

    async def test_handoff_returns_configured_message_in_detected_language(self):
        run = await _run_flow(decisions=[_handoff("de")], config=self.CONFIG)

        self.assertTrue(run.result["ok"])
        self.assertEqual(run.result["branch"], "handoff")
        self.assertEqual(run.result["lang"], "de")
        self.assertEqual(run.result["reply_text"], "Wir melden uns.")

    async def test_handoff_falls_back_to_english_for_unconfigured_language(self):
        run = await _run_flow(decisions=[_handoff("fr")], config=self.CONFIG)

        self.assertEqual(run.result["reply_text"], "We will reply.")

    async def test_handoff_after_a_section_request(self):
        run = await _run_flow(decisions=[_request_sections(1), _handoff("en")], config=self.CONFIG)

        self.assertEqual(run.result["branch"], "handoff")
        self.assertEqual(run.result["reply_text"], "We will reply.")
        self.assertEqual(len(run.decider.calls), 2)

    async def test_handoff_turn_is_persisted_like_any_other(self):
        run = await _run_flow(decisions=[_handoff("en")], config=self.CONFIG)

        self.assertEqual(
            run.save_turn.await_args_list,
            [call("biz-1#user-1", "user", "Hi"), call("biz-1#user-1", "assistant", "We will reply.")],
        )


class LengthRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_overlong_answer_is_shortened_once(self):
        run = await _run_flow(decisions=[_answer("x" * 1200)], repaired=["y" * 500])

        self.assertEqual(run.result["reply_text"], "y" * 500)
        run.repair.assert_awaited_once()
        self.assertEqual(run.repair.await_args.kwargs["retry_count"], 1)
        self.assertEqual(run.repair.await_args.kwargs["message_text"], "Hi")
        self.assertEqual(run.repair.await_args.kwargs["max_reply_chars"], 1000)
        self.assertEqual(run.save_turn.await_args_list[1], call("biz-1#user-1", "assistant", "y" * 500))

    async def test_answer_still_too_long_after_repair_is_truncated(self):
        run = await _run_flow(decisions=[_answer("x" * 1200)], repaired=["z" * 1100])

        run.repair.assert_awaited_once()
        self.assertEqual(run.result["reply_text"], "z" * 997 + "...")
        self.assertEqual(len(run.result["reply_text"]), 1000)

    async def test_answer_within_limit_is_not_repaired(self):
        run = await _run_flow(decisions=[_answer("x" * 1000)])

        run.repair.assert_not_called()
        self.assertEqual(run.result["reply_text"], "x" * 1000)

    async def test_whatsapp_uses_the_shorter_limit(self):
        run = await _run_flow({"channel": "whatsapp"}, decisions=[_answer("x" * 600)], repaired=["y" * 400])

        self.assertEqual(run.repair.await_args.kwargs["max_reply_chars"], 500)
        self.assertEqual(run.result["reply_text"], "y" * 400)


class FirstReplyAddendumTests(unittest.IsolatedAsyncioTestCase):
    CONFIG = {**BASE_CONFIG, "first_reply_addendum": "  Before we start\n\nPrivacy policy  "}

    async def test_first_whatsapp_reply_gets_the_configured_addendum_only(self):
        run = await _run_flow({"channel": "whatsapp"}, decisions=[_answer()], config=self.CONFIG)

        self.assertEqual(run.result["additional_reply"], "Before we start\n\nPrivacy policy")

    async def test_whatsapp_without_configured_addendum_adds_nothing(self):
        run = await _run_flow({"channel": "whatsapp"}, decisions=[_answer()])

        self.assertEqual(run.result["additional_reply"], "")

    async def test_addendum_is_not_repeated_in_an_ongoing_chat(self):
        history = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]
        run = await _run_flow({"channel": "whatsapp"}, decisions=[_answer()], config=self.CONFIG, history=history)

        self.assertEqual(run.result["additional_reply"], "")

    async def test_other_channels_never_get_the_addendum(self):
        for channel in ("widget", "telegram"):
            with self.subTest(channel=channel):
                run = await _run_flow({"channel": channel}, decisions=[_answer()], config=self.CONFIG)

                self.assertEqual(run.result["additional_reply"], "")


class ErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_business_config_returns_error_without_calling_the_model(self):
        run = await _run_flow(decisions=[], config=None)

        self.assertFalse(run.result["ok"])
        self.assertEqual(run.result["branch"], "error")
        self.assertEqual(run.result["reply_text"], "Business configuration was not found.")
        self.assertEqual(run.result["number_images"], 0)
        self.assertEqual(run.decider.calls, [])
        run.save_turn.assert_not_called()

    async def test_blank_system_prompt_is_treated_as_missing_config(self):
        run = await _run_flow(decisions=[], config={"system_prompt": "   "})

        self.assertFalse(run.result["ok"])
        self.assertEqual(run.result["branch"], "error")
        self.assertEqual(run.decider.calls, [])

    async def test_history_persistence_failure_does_not_lose_the_reply(self):
        with self.assertLogs(MODULE, level="ERROR") as logs:
            run = await _run_flow(decisions=[_answer("Still delivered.")], save_turn_error=RuntimeError("down"))

        self.assertTrue(run.result["ok"])
        self.assertEqual(run.result["reply_text"], "Still delivered.")
        self.assertIn("History persistence failed", logs.output[0])


if __name__ == "__main__":
    unittest.main()
