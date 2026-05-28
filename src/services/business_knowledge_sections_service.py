from src.repository import business_knowledge_sections_repository


def get_section_options(business_id: str) -> list[dict]:
    items = business_knowledge_sections_repository.query_section_options_by_business_id(business_id)
    options: list[dict] = []
    for number, item in enumerate(items, start=1):
        section_id = item.get("section_id")
        title = item.get("title")
        if isinstance(section_id, str) and section_id.strip() and isinstance(title, str) and title.strip():
            options.append(
                {
                    "number": number,
                    "section_id": section_id.strip(),
                    "title": title.strip(),
                }
            )
    return options


def map_section_ids_by_numbers(section_options: list[dict], selected_numbers: list[int]) -> list[str]:
    section_by_number = {
        option["number"]: option["section_id"]
        for option in section_options
        if isinstance(option.get("number"), int) and isinstance(option.get("section_id"), str)
    }
    section_ids: list[str] = []
    seen: set[str] = set()
    for number in selected_numbers:
        section_id = section_by_number.get(number)
        if section_id and section_id not in seen:
            seen.add(section_id)
            section_ids.append(section_id)
    return section_ids


def get_sections_map_by_ids(business_id: str, section_ids: list[str]) -> dict[str, dict]:
    items = business_knowledge_sections_repository.get_sections_by_ids(business_id, section_ids)
    sections: dict[str, dict] = {}
    for item in items:
        section_id = item.get("section_id")
        title = item.get("title")
        text = item.get("text")
        if (
            isinstance(section_id, str)
            and section_id.strip()
            and isinstance(title, str)
            and title.strip()
            and isinstance(text, str)
            and text.strip()
        ):
            sections[section_id.strip()] = {"title": title.strip(), "text": text.strip()}
    return sections


def filter_sections(sections_map: dict[str, dict], selected_ids: list[str]) -> list[dict]:
    filtered: list[dict] = []
    for section_id in selected_ids:
        section = sections_map.get(section_id)
        if section:
            filtered.append(section)
    return filtered
