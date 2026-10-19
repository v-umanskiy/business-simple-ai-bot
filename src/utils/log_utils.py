from typing import Any


def chars(value: Any) -> int:
    """Length of user or model text, so logs carry sizes but never content."""
    return len(str(value or ""))
