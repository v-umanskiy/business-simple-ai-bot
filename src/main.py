from fastapi import FastAPI, Header, HTTPException, Request
from mangum import Mangum

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
    message_text = payload.get("message_text")
    if not message_text:
        raise HTTPException(status_code=400, detail="message_text is required")

    reply_text = await generate_reply(message_text)

    return {
        "ok": True,
        "branch": "reply_text",
        "reply_text": reply_text,
        "echo": payload,
    }
