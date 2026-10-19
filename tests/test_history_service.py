import os
import unittest
from unittest.mock import AsyncMock, patch

from src.repository import history_repository
from src.services import history_service


class SaveTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_is_stored_with_millisecond_timestamp(self):
        with (
            patch.object(history_repository, "save_turn_item", AsyncMock(return_value="saved")) as save,
            patch("src.services.history_service.time.time", return_value=1_700_000_000.123),
        ):
            result = await history_service.save_turn("biz#user", "user", "Hello")

        self.assertEqual(result, "saved")
        save.assert_awaited_once_with(
            {"chat_key": "biz#user", "ts": 1_700_000_000_123, "role": "user", "text": "Hello"}
        )


class LoadModelHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_turns_are_mapped_to_model_roles_and_empty_text_is_skipped(self):
        turns = [
            {"role": "user", "text": "Hi"},
            {"role": "assistant", "text": "Hello"},
            {"role": "assistant", "text": ""},
            {"role": "something-else", "text": "Odd role"},
            {"role": "user"},
        ]
        with patch.object(history_repository, "query_turns", AsyncMock(return_value=turns)):
            history = await history_service.load_model_history("biz#user")

        self.assertEqual(
            history,
            [
                {"role": "user", "text": "Hi"},
                {"role": "assistant", "text": "Hello"},
                {"role": "user", "text": "Odd role"},
            ],
        )

    async def test_window_defaults_to_twelve_turns(self):
        with (
            patch.dict(os.environ, clear=False),
            patch.object(history_repository, "query_turns", AsyncMock(return_value=[])) as query,
        ):
            os.environ.pop("WINDOW_TURNS", None)
            await history_service.load_model_history("biz#user")

        query.assert_awaited_once_with("biz#user", 12)

    async def test_window_can_be_configured_through_the_environment(self):
        with (
            patch.dict(os.environ, {"WINDOW_TURNS": "5"}),
            patch.object(history_repository, "query_turns", AsyncMock(return_value=[])) as query,
        ):
            await history_service.load_model_history("biz#user")

        query.assert_awaited_once_with("biz#user", 5)


class PruneHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_prune_keeps_thirty_turns_by_default_and_returns_deleted_count(self):
        with (
            patch.dict(os.environ, clear=False),
            patch.object(history_repository, "delete_older_turns", AsyncMock(return_value=4)) as delete,
        ):
            os.environ.pop("WINDOW_TURNS", None)
            deleted = await history_service.prune_history("biz#user")

        self.assertEqual(deleted, 4)
        delete.assert_awaited_once_with("biz#user", 30)

    async def test_prune_window_can_be_configured_through_the_environment(self):
        with (
            patch.dict(os.environ, {"WINDOW_TURNS": "8"}),
            patch.object(history_repository, "delete_older_turns", AsyncMock(return_value=0)) as delete,
        ):
            await history_service.prune_history("biz#user")

        delete.assert_awaited_once_with("biz#user", 8)


if __name__ == "__main__":
    unittest.main()
