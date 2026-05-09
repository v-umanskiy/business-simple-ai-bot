import os
import boto3
from boto3.dynamodb.conditions import Key

_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(os.getenv("IMAGE_CATALOG_TABLE_NAME"))


def query_images_by_business_id(business_id: str) -> list[dict]:
    response = _table.query(
        KeyConditionExpression=Key("business_id").eq(business_id),
        ScanIndexForward=True,
    )
    return response.get("Items", [])
