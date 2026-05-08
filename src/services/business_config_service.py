from src.repository import business_config_repository


def get_business_config(business_id: str) -> dict | None:
    return business_config_repository.get_by_business_id(business_id)


def get_system_prompt(business_id: str) -> str | None:
    item = get_business_config(business_id)
    if not item:
        return None
    return item.get("system_prompt")
