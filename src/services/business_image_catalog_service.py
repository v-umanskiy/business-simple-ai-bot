from src.repository import business_image_catalog_repository


def get_image_url_map(business_id: str) -> dict[str, str]:
    items = business_image_catalog_repository.query_images_by_business_id(business_id)
    mapping: dict[str, str] = {}
    for item in items:
        image_id = item.get("image_id")
        url = item.get("url")
        if isinstance(image_id, str) and image_id.strip() and isinstance(url, str) and url.strip():
            mapping[image_id.strip()] = url.strip()
    return mapping


def get_image_ids(business_id: str) -> list[str]:
    return list(get_image_url_map(business_id).keys())
