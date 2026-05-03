from fastapi import FastAPI, Header, HTTPException, Request
from mangum import Mangum

from src.services.business_config_service import get_system_prompt
from src.services.history_service import load_model_history, prune_history, save_turn
from src.services.secret_service import get_secret
from src.services.text_generation_service import generate_reply

app = FastAPI()
handler = Mangum(app)


@app.get("/")
def read_root():
    return {"ok": True, "service": "business-simple-ai-bot"}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: str | None = Header(None),
):
    expected_secret = get_secret("WEBHOOK_SECRET")
    if expected_secret and x_webhook_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Bad secret")

    payload = await request.json()
    business_id = payload.get("business_id")
    user_id = payload.get("user_id")
    message_text = payload.get("message_text")
    if not business_id:
        raise HTTPException(status_code=400, detail="business_id is required")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not message_text:
        raise HTTPException(status_code=400, detail="message_text is required")

    system_prompt = get_system_prompt(business_id)
    if not system_prompt:
        return {
            "ok": False,
            "branch": "error",
            "reply_text": "Business configuration was not found.",
            "echo": payload,
        }

    chat_key = f"{business_id}#{user_id}"
    history = await load_model_history(chat_key)
    reply_text = await generate_reply(message_text, system_prompt, history)
    await save_turn(chat_key, "user", message_text)
    await save_turn(chat_key, "assistant", reply_text)
    await prune_history(chat_key)

    return {
        "ok": True,
        "branch": "reply_text",
        "reply_text": reply_text,
        "echo": payload,
    }
