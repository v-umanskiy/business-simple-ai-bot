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
