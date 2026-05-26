from dataclasses import dataclass
from typing import Any


CHANNEL_REPLY_LIMITS = {
    "widget": 1000,
    "telegram": 1000,
    "whatsapp": 500,
}
DEFAULT_CHANNEL = "widget"


@dataclass(frozen=True)
class ChannelPolicy:
    channel: str
    max_reply_chars: int
    raw_channel: Any
    used_default: bool


def resolve_channel_policy(payload: dict[str, Any]) -> ChannelPolicy:
    raw_channel = payload.get("channel", payload.get("platform", payload.get("plattform")))
    channel = str(raw_channel or "").strip().lower()
    if channel in CHANNEL_REPLY_LIMITS:
        return ChannelPolicy(
            channel=channel,
            max_reply_chars=CHANNEL_REPLY_LIMITS[channel],
            raw_channel=raw_channel,
            used_default=False,
        )

    return ChannelPolicy(
        channel=DEFAULT_CHANNEL,
        max_reply_chars=CHANNEL_REPLY_LIMITS[DEFAULT_CHANNEL],
        raw_channel=raw_channel,
        used_default=bool(raw_channel),
    )
