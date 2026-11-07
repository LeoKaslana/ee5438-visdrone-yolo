"""Checks for the diagnostic matching used in failure-case selection."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_failures import area_group, iou_xywh, match_image


class FailureAnalysisTests(unittest.TestCase):
    def test_iou_and_size_boundaries(self) -> None:
        self.assertEqual(iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou_xywh([0, 0, 10, 10], [20, 20, 10, 10]), 0.0)
        self.assertEqual(area_group([0, 0, 32, 32]), "medium")
        self.assertEqual(area_group([0, 0, 96, 96]), "large")

    def test_one_prediction_matches_one_same_class_object(self) -> None:
        annotations = [
            {"id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 3, "category_id": 2, "bbox": [0, 0, 10, 10]},
        ]
        predictions = [{"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
        self.assertEqual(len(match_image(annotations, predictions, 0.25)), 1)
        self.assertTrue(match_image(annotations, predictions, 0.25).issubset({1, 2}))


if __name__ == "__main__":
    unittest.main()
