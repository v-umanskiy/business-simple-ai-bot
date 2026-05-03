from src.repository import business_config_repository


def get_system_prompt(business_id: str) -> str | None:
    item = business_config_repository.get_by_business_id(business_id)
    if not item:
        return None
    return item.get("system_prompt")
