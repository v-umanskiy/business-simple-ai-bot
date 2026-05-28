import os
import boto3
from boto3.dynamodb.conditions import Key

_dynamodb = boto3.resource("dynamodb")
_table_name = os.getenv("KNOWLEDGE_SECTIONS_TABLE_NAME")
_table = _dynamodb.Table(_table_name)


def query_section_options_by_business_id(business_id: str) -> list[dict]:
    options: list[dict] = []
    query_kwargs = {
        "KeyConditionExpression": Key("business_id").eq(business_id),
        "ScanIndexForward": True,
        "ProjectionExpression": "section_id, title",
    }

    while True:
        response = _table.query(**query_kwargs)
        for item in response.get("Items", []):
            section_id = item.get("section_id")
            title = item.get("title")
            if isinstance(section_id, str) and section_id.strip() and isinstance(title, str) and title.strip():
                options.append({"section_id": section_id.strip(), "title": title.strip()})

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            return options
        query_kwargs["ExclusiveStartKey"] = last_key


def get_sections_by_ids(business_id: str, section_ids: list[str]) -> list[dict]:
    if not _table_name or not section_ids:
        return []

    unique_ids: list[str] = []
    seen: set[str] = set()
    for section_id in section_ids:
        normalized = str(section_id).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_ids.append(normalized)

    if not unique_ids:
        return []

    keys = [{"business_id": business_id, "section_id": section_id} for section_id in unique_ids]

    response = _dynamodb.meta.client.batch_get_item(
        RequestItems={
            _table_name: {
                "Keys": keys,
                "ProjectionExpression": "section_id, title, #txt",
                "ExpressionAttributeNames": {"#txt": "text"},
            }
        }
    )
    return response.get("Responses", {}).get(_table_name, [])
