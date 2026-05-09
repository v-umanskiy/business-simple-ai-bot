from src.repository import business_knowledge_sections_repository


def get_sections_map(business_id: str) -> dict[str, str]:
    items = business_knowledge_sections_repository.query_sections_by_business_id(business_id)
    sections: dict[str, str] = {}
    for item in items:
        section_name = item.get("section_name")
        text = item.get("text")
        if isinstance(section_name, str) and section_name.strip() and isinstance(text, str) and text.strip():
            sections[section_name.strip()] = text.strip()
    return sections


def filter_sections(sections_map: dict[str, str], selected_names: list[str]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for name in selected_names:
        if name in sections_map:
            filtered[name] = sections_map[name]
    return filtered
