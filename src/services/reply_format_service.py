def normalize_plain_text_reply(reply_text: str) -> str:
    text = str(reply_text or "")
    return text.replace("**", "").strip()
