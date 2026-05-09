import os
import boto3
from boto3.dynamodb.conditions import Key

_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(os.getenv("KNOWLEDGE_SECTIONS_TABLE_NAME"))
_table_name = os.getenv("KNOWLEDGE_SECTIONS_TABLE_NAME")


def query_section_names_by_business_id(business_id: str) -> list[str]:
    response = _table.query(
        KeyConditionExpression=Key("business_id").eq(business_id),
        ScanIndexForward=True,
        ProjectionExpression="section_name",
    )
    names: list[str] = []
    for item in response.get("Items", []):
        name = item.get("section_name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def get_sections_by_names(business_id: str, section_names: list[str]) -> list[dict]:
    if not _table_name or not section_names:
        return []

    unique_names: list[str] = []
    seen: set[str] = set()
    for name in section_names:
        normalized = str(name).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_names.append(normalized)

    if not unique_names:
        return []

    keys = [{"business_id": business_id, "section_name": name} for name in unique_names]

    response = _dynamodb.meta.client.batch_get_item(
        RequestItems={
            _table_name: {
                "Keys": keys,
                "ProjectionExpression": "section_name, #txt",
                "ExpressionAttributeNames": {"#txt": "text"},
            }
        }
    )
    return response.get("Responses", {}).get(_table_name, [])
