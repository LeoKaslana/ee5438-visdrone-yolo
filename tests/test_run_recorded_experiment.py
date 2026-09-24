"""Tests for experiment metadata and result summaries (no training required)."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_recorded_experiment import build_training_command, save_manifest, summarize_results


class RecordedExperimentTests(unittest.TestCase):
    def test_summarize_results_keeps_last_and_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "epoch,metrics/mAP50-95(B)\n1,0.15\n2,0.12\n3,0.18\n",
                encoding="utf-8",
            )
            result = summarize_results(path)
            self.assertEqual(result["completed_epochs"], 3)
            self.assertEqual(result["last_epoch"]["epoch"], "3")
            self.assertEqual(result["best_builtin_epoch"]["epoch"], "3")

    def test_training_command_contains_fixed_comparison_settings(self) -> None:
        args = argparse.Namespace(variant="p2", epochs=50, batch=8, imgsz=640, workers=0, seed=0, fraction=1.0, save_period=10)
        command = build_training_command(args, "strict-test-p2")
        self.assertIn("train_p2.py", command[1])
        self.assertEqual(command[command.index("--batch") + 1], "8")
        self.assertEqual(command[command.index("--workers") + 1], "0")
        self.assertEqual(command[command.index("--save-period") + 1], "10")

    def test_manifest_write_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_manifest(path, {"status": "training", "run_name": "sample"})
            self.assertIn('"status": "training"', path.read_text(encoding="utf-8"))
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
