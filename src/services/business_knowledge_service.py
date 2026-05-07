from src.repository import business_knowledge_repository


def get_sections_map(business_id: str) -> dict[str, str]:
    item = business_knowledge_repository.get_by_business_id(business_id)
    if not item:
        return {}

    sections = item.get("sections")
    if not isinstance(sections, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in sections.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    return normalized


def filter_sections(sections_map: dict[str, str], selected_names: list[str]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for name in selected_names:
        if name in sections_map:
            filtered[name] = sections_map[name]
    return filtered
