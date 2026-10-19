import unittest
from unittest.mock import patch

from src.repository import (
    business_config_repository,
    business_image_catalog_repository,
    business_knowledge_sections_repository,
)
from src.services import business_config_service, business_image_catalog_service
from src.services.business_knowledge_sections_service import (
    filter_sections,
    get_section_options,
    get_sections_map_by_ids,
    map_section_ids_by_numbers,
)


class BusinessConfigServiceTests(unittest.TestCase):
    def test_system_prompt_is_read_from_the_config_item(self):
        with patch.object(business_config_repository, "get_by_business_id", return_value={"system_prompt": "Hi"}):
            self.assertEqual(business_config_service.get_system_prompt("biz"), "Hi")

    def test_unknown_business_has_no_config_and_no_prompt(self):
        with patch.object(business_config_repository, "get_by_business_id", return_value=None):
            self.assertIsNone(business_config_service.get_business_config("biz"))
            self.assertIsNone(business_config_service.get_system_prompt("biz"))


class ImageCatalogServiceTests(unittest.TestCase):
    ITEMS = [
        {"image_id": " a ", "url": " https://img.test/a.png "},
        {"image_id": "b", "url": "https://img.test/b.png"},
        {"image_id": "", "url": "https://img.test/blank-id.png"},
        {"image_id": "no-url"},
        {"image_id": "blank-url", "url": "   "},
        {"image_id": 7, "url": "https://img.test/numeric-id.png"},
    ]

    def test_valid_items_are_trimmed_and_invalid_ones_skipped(self):
        with patch.object(business_image_catalog_repository, "query_images_by_business_id", return_value=self.ITEMS):
            mapping = business_image_catalog_service.get_image_url_map("biz")

        self.assertEqual(mapping, {"a": "https://img.test/a.png", "b": "https://img.test/b.png"})

    def test_image_ids_follow_catalog_order(self):
        with patch.object(business_image_catalog_repository, "query_images_by_business_id", return_value=self.ITEMS):
            self.assertEqual(business_image_catalog_service.get_image_ids("biz"), ["a", "b"])


class SectionOptionsTests(unittest.TestCase):
    def test_options_are_numbered_from_one_and_trimmed(self):
        items = [
            {"section_id": " s1 ", "title": " Prices "},
            {"section_id": "s2", "title": "Hours"},
        ]
        with patch.object(
            business_knowledge_sections_repository, "query_section_options_by_business_id", return_value=items
        ):
            options = get_section_options("biz")

        self.assertEqual(
            options,
            [
                {"number": 1, "section_id": "s1", "title": "Prices"},
                {"number": 2, "section_id": "s2", "title": "Hours"},
            ],
        )

    def test_items_without_id_or_title_are_not_offered(self):
        items = [
            {"section_id": "s1", "title": "Prices"},
            {"section_id": "", "title": "No id"},
            {"section_id": "s3"},
            {"section_id": "s4", "title": "   "},
        ]
        with patch.object(
            business_knowledge_sections_repository, "query_section_options_by_business_id", return_value=items
        ):
            options = get_section_options("biz")

        self.assertEqual([o["section_id"] for o in options], ["s1"])


class MapSectionIdsByNumbersTests(unittest.TestCase):
    OPTIONS = [
        {"number": 1, "section_id": "s1", "title": "A"},
        {"number": 2, "section_id": "s2", "title": "B"},
        {"number": 3, "section_id": "s3", "title": "C"},
    ]

    def test_numbers_are_mapped_in_requested_order(self):
        self.assertEqual(map_section_ids_by_numbers(self.OPTIONS, [3, 1]), ["s3", "s1"])

    def test_duplicates_and_unknown_numbers_are_dropped(self):
        self.assertEqual(map_section_ids_by_numbers(self.OPTIONS, [2, 2, 99, 0]), ["s2"])

    def test_empty_request_maps_to_nothing(self):
        self.assertEqual(map_section_ids_by_numbers(self.OPTIONS, []), [])


class SectionContentTests(unittest.TestCase):
    def test_sections_are_keyed_by_id_and_incomplete_items_are_dropped(self):
        items = [
            {"section_id": " s1 ", "title": " Prices ", "text": " 100 EUR "},
            {"section_id": "s2", "title": "No text"},
            {"section_id": "s3", "title": "Blank text", "text": "  "},
            {"title": "No id", "text": "x"},
        ]
        with patch.object(business_knowledge_sections_repository, "get_sections_by_ids", return_value=items) as fetch:
            sections = get_sections_map_by_ids("biz", ["s1", "s2", "s3"])

        fetch.assert_called_once_with("biz", ["s1", "s2", "s3"])
        self.assertEqual(sections, {"s1": {"title": "Prices", "text": "100 EUR"}})

    def test_filter_keeps_selection_order_and_skips_missing_ids(self):
        sections_map = {"s1": {"title": "A", "text": "a"}, "s2": {"title": "B", "text": "b"}}

        self.assertEqual(
            filter_sections(sections_map, ["s2", "missing", "s1"]),
            [{"title": "B", "text": "b"}, {"title": "A", "text": "a"}],
        )


if __name__ == "__main__":
    unittest.main()
