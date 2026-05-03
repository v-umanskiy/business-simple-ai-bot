# business-simple-ai-bot

Minimal AWS Lambda (FastAPI + Mangum) for chatbot API integration tests.

## Endpoints

- `GET /` health check
- `POST /webhook` validates `X-Webhook-Secret` and returns test JSON

## Environment

- `SECRETS_ARN` (optional): ARN/name of Secrets Manager secret
- Secret payload key: `WEBHOOK_SECRET`

Default secret id fallback: `BUSINESS_SIMPLE_AI_BOT_SECRET`.

## Local contract (test)

`POST /webhook` returns:

```json
{
  "ok": true,
  "branch": "reply_text",
  "reply_text": "Connection works.",
  "echo": {"...": "incoming payload"}
}
```
