import os

import boto3

_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(os.getenv("BUSINESS_KNOWLEDGE_TABLE_NAME"))


def get_by_business_id(business_id: str) -> dict | None:
    response = _table.get_item(Key={"business_id": business_id})
    return response.get("Item")
