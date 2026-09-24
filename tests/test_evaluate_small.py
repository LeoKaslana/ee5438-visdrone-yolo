"""Small checks for the original-pixel COCO conversion."""

import unittest
from types import SimpleNamespace

import numpy as np

from src.evaluate_small import SMALL_AREA, mean_valid, overall_metrics, yolo_box_to_xywh


class EvaluateSmallTests(unittest.TestCase):
    def test_normalized_box_uses_original_image_size(self) -> None:
        bbox, clipped = yolo_box_to_xywh([0.5, 0.5, 0.1, 0.1], 1000, 500)
        self.assertEqual(bbox, [450.0, 225.0, 100.0, 50.0])
        self.assertFalse(clipped)
        self.assertGreater(bbox[2] * bbox[3], SMALL_AREA)

    def test_boundary_box_is_clipped(self) -> None:
        bbox, clipped = yolo_box_to_xywh([0.99, 0.5, 0.1, 0.2], 100, 100)
        self.assertAlmostEqual(bbox[0], 94.0)
        self.assertAlmostEqual(bbox[2], 6.0)
        self.assertTrue(clipped)

    def test_mean_valid_ignores_missing_coco_slices(self) -> None:
        self.assertAlmostEqual(mean_valid(np.array([-1.0, 0.2, 0.4])), 0.3)
        self.assertIsNone(mean_valid(np.array([-1.0, -1.0])))

    def test_overall_ap_uses_configured_high_max_det(self) -> None:
        precision = np.full((2, 3, 1, 4, 3), -1.0)
        precision[:, :, 0, 0, -1] = 0.2
        precision[:, :, 0, 1, -1] = 0.1
        evaluation = SimpleNamespace(
            eval={"precision": precision},
            params=SimpleNamespace(
                areaRngLbl=["all", "small", "medium", "large"],
                iouThrs=np.array([0.5, 0.75]),
            ),
        )
        metrics = overall_metrics(evaluation)
        self.assertAlmostEqual(metrics["AP"], 0.2)
        self.assertAlmostEqual(metrics["AP_small"], 0.1)


if __name__ == "__main__":
    unittest.main()
