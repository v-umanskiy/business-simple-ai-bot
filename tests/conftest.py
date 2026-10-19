import os

# Repository modules create boto3 DynamoDB resources at import time. Dummy table
# names, region and credentials let them import without ever reaching real AWS.
os.environ.pop("AWS_PROFILE", None)
os.environ.update(
    {
        "AWS_DEFAULT_REGION": "eu-central-1",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "BUSINESS_CONFIG_TABLE_NAME": "test-business-config",
        "IMAGE_CATALOG_TABLE_NAME": "test-image-catalog",
        "KNOWLEDGE_SECTIONS_TABLE_NAME": "test-knowledge-sections",
        "HISTORY_TABLE_NAME": "test-history",
    }
)
