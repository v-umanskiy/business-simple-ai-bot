from src.repository import business_knowledge_sections_repository


def get_section_names(business_id: str) -> list[str]:
    return business_knowledge_sections_repository.query_section_names_by_business_id(business_id)


def get_sections_map_by_names(business_id: str, section_names: list[str]) -> dict[str, str]:
    items = business_knowledge_sections_repository.get_sections_by_names(business_id, section_names)
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
