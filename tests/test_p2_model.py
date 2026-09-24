"""Check the project's P2/P3/P4 topology before a long experiment."""

import sys
import unittest
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class P2ModelTests(unittest.TestCase):
    def test_detection_head_has_p2_p3_p4_strides(self) -> None:
        model = YOLO(str(ROOT / "configs" / "yolo11n-p2p3p4.yaml"))
        self.assertEqual(model.model.stride.tolist(), [4.0, 8.0, 16.0])
        self.assertEqual(model.model.model[-1].nc, 10)
        self.assertEqual(model.model.model[-1].f, [19, 22, 25])


if __name__ == "__main__":
    unittest.main()
