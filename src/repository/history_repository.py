import asyncio
import os

import boto3
from boto3.dynamodb.conditions import Key

_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(os.getenv("HISTORY_TABLE_NAME"))


async def save_turn_item(item: dict):
    return await asyncio.to_thread(_table.put_item, Item=item)


async def query_turns(chat_key: str, limit: int):
    def _query():
        response = _table.query(
            KeyConditionExpression=Key("chat_key").eq(chat_key),
            ScanIndexForward=False,
            Limit=limit,
        )
        return list(reversed(response.get("Items", [])))

    return await asyncio.to_thread(_query)


async def delete_older_turns(chat_key: str, keep: int) -> int:
    def _cleanup() -> int:
        response = _table.query(
            KeyConditionExpression=Key("chat_key").eq(chat_key),
            ScanIndexForward=True,
        )
        items = response.get("Items", [])
        to_delete = items[:-keep] if keep > 0 else items
        if not to_delete:
            return 0

        with _table.batch_writer() as batch:
            for item in to_delete:
                batch.delete_item(Key={"chat_key": item["chat_key"], "ts": item["ts"]})
        return len(to_delete)

    return await asyncio.to_thread(_cleanup)
