"""Basic checks for concise run results."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_experiment import build_train_command, save_json, training_metrics


class RunExperimentTests(unittest.TestCase):
    def test_training_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "results.csv"
            csv_path.write_text(
                "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.2,0.1\n2,0.3,0.15\n",
                encoding="utf-8",
            )
            result = training_metrics(csv_path)
            self.assertEqual(result["completed_epochs"], 2)
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["best_mAP50_95"], 0.15)

    def test_command_uses_fixed_batch_and_workers(self) -> None:
        args = argparse.Namespace(variant="p2", epochs=50, batch=8, workers=0, imgsz=640, seed=0, fraction=1.0, save_period=10)
        command = build_train_command(args, "strict-test-p2")
        self.assertEqual(command[command.index("--batch") + 1], "8")
        self.assertEqual(command[command.index("--workers") + 1], "0")

    def test_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_json(path, {"status": "complete"})
            self.assertIn('"status": "complete"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
