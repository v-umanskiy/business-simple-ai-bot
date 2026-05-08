from src.utils.constants import DEMO_HANDOFF_MESSAGES, REAL_HANDOFF_MESSAGES


def _normalize_lang(lang: str | None) -> str:
    if not lang:
        return "en"
    normalized = str(lang).strip().lower()
    if not normalized:
        return "en"
    return normalized.split("-")[0]


def _pick_by_lang(message_map: dict[str, str], lang: str) -> str:
    if lang in message_map and message_map[lang].strip():
        return message_map[lang].strip()
    if "en" in message_map and message_map["en"].strip():
        return message_map["en"].strip()
    return ""


def resolve_handoff_message(config: dict, lang: str | None) -> str:
    normalized_lang = _normalize_lang(lang)

    configured = config.get("handoff_message") if isinstance(config, dict) else None
    if isinstance(configured, dict):
        configured_map: dict[str, str] = {}
        for key, value in configured.items():
            if isinstance(key, str) and isinstance(value, str):
                configured_map[key.strip().lower()] = value
        selected = _pick_by_lang(configured_map, normalized_lang)
        if selected:
            return selected

    is_demo = bool(config.get("demo")) if isinstance(config, dict) else False
    defaults = DEMO_HANDOFF_MESSAGES if is_demo else REAL_HANDOFF_MESSAGES
    return _pick_by_lang(defaults, normalized_lang)
