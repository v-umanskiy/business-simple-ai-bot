import os
import time

from src.repository import history_repository


async def save_turn(chat_key: str, role: str, text: str):
    item = {
        "chat_key": chat_key,
        "ts": int(time.time() * 1000),
        "role": role,
        "text": text,
    }
    return await history_repository.save_turn_item(item)


async def load_model_history(chat_key: str) -> list[dict]:
    window_turns = int(os.getenv("WINDOW_TURNS", "12"))
    turns = await history_repository.query_turns(chat_key, window_turns)

    history = []
    for turn in turns:
        text = turn.get("text")
        if not text:
            continue

        role = "assistant" if turn.get("role") == "assistant" else "user"
        history.append({"role": role, "text": text})

    return history
