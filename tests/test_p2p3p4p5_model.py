"""Check the four-scale topology before a long experiment."""

import unittest
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


class FourScaleModelTests(unittest.TestCase):
    def test_detection_head_has_four_strides(self) -> None:
        model = YOLO(str(ROOT / "configs" / "yolo11n-p2p3p4p5.yaml"))
        self.assertEqual(model.model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])
        self.assertEqual(model.model.model[-1].nc, 10)
        self.assertEqual(model.model.model[-1].f, [19, 22, 25, 28])


if __name__ == "__main__":
    unittest.main()
