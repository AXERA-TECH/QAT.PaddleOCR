import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools/data/convert_hiertext.py"
SPEC = importlib.util.spec_from_file_location("convert_hiertext", SCRIPT)
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


class HierTextConversionTest(unittest.TestCase):
    def test_line_entries_preserve_line_polygon_and_text(self):
        annotation = {
            "paragraphs": [
                {
                    "legible": True,
                    "lines": [
                        {
                            "vertices": [[1, 2], [9, 2], [9, 6], [1, 6]],
                            "text": "two words",
                            "legible": True,
                            "words": [
                                {"text": "two", "legible": True},
                                {"text": "words", "legible": True},
                            ],
                        }
                    ],
                }
            ]
        }

        entries = CONVERTER.collect_det_entries(annotation, "line")

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["transcription"], "two words")
        self.assertEqual(
            entries[0]["points"],
            [[1.0, 2.0], [9.0, 2.0], [9.0, 6.0], [1.0, 6.0]],
        )

    def test_illegible_paragraph_makes_line_ignored(self):
        annotation = {
            "paragraphs": [
                {
                    "legible": False,
                    "lines": [
                        {
                            "vertices": [[0, 0], [4, 0], [4, 2], [0, 2]],
                            "text": "unreadable",
                            "legible": True,
                        }
                    ],
                }
            ]
        }

        entries = CONVERTER.collect_det_entries(annotation, "line")

        self.assertEqual(entries[0]["transcription"], "###")

    def test_word_level_remains_available(self):
        annotation = {
            "paragraphs": [
                {
                    "lines": [
                        {
                            "words": [
                                {
                                    "vertices": [[0, 0], [3, 0], [3, 2], [0, 2]],
                                    "text": "word",
                                    "legible": True,
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        entries = CONVERTER.collect_det_entries(annotation, "word")

        self.assertEqual(entries[0]["transcription"], "word")


if __name__ == "__main__":
    unittest.main()
