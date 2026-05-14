import logging
import os

from fastapi import FastAPI, Header, HTTPException, Request
from mangum import Mangum

from src.services.secret_service import get_secret
from src.services.bot_graph_service import run_bot_flow

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger().setLevel(LOG_LEVEL)

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

    return await run_bot_flow(payload)
