"""Run one experiment with native terminal progress and a concise result record."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC_KEY = "metrics/mAP50-95(B)"


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"completed_epochs": 0}
    best = max(rows, key=lambda row: float(row[METRIC_KEY]))
    return {
        "completed_epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_mAP50": float(best["metrics/mAP50(B)"]),
        "best_mAP50_95": float(best[METRIC_KEY]),
        "last_epoch": int(rows[-1]["epoch"]),
    }


def build_train_command(args: argparse.Namespace, run_name: str) -> list[str]:
    script = ROOT / "src" / ("train_baseline.py" if args.variant == "baseline" else "train_p2.py")
    return [
        sys.executable, str(script),
        "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--imgsz", str(args.imgsz), "--workers", str(args.workers),
        "--seed", str(args.seed), "--fraction", str(args.fraction),
        "--save-period", str(args.save_period), "--name", run_name,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "p2"), required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--no-eval", action="store_true", help="Skip AP-small evaluation for a smoke test")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.comparison_id):
        parser.error("Invalid comparison ID")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.imgsz < 32 or not 0 < args.fraction <= 1:
        parser.error("Invalid training settings")
    if args.save_period == 0 or args.eval_batch < 1:
        parser.error("Invalid checkpoint or evaluation batch setting")

    run_name = f"{args.comparison_id}-{args.variant}"
    train_dir = ROOT / "runs" / args.variant / run_name
    evaluation_dir = ROOT / "runs" / "evaluation" / f"{run_name}-val"
    record_path = ROOT / "runs" / "records" / f"{run_name}.json"
    if any(path.exists() for path in (train_dir, evaluation_dir, record_path)):
        parser.error(f"Run already exists: {run_name}; choose a new comparison ID")
    if not (ROOT / "data" / "VisDrone" / "VisDrone-local.yaml").is_file():
        parser.error("VisDrone dataset is not prepared")
    if not (ROOT / "yolo11n.pt").is_file():
        parser.error("Missing yolo11n.pt")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    record = {
        "run": run_name,
        "variant": args.variant,
        "status": "training",
        "started_at": local_now(),
        "parameters": {
            "epochs": args.epochs, "batch": args.batch, "workers": args.workers,
            "imgsz": args.imgsz, "seed": args.seed, "fraction": args.fraction,
            "save_period": args.save_period, "eval_batch": args.eval_batch,
        },
        "git_commit": commit,
        "training_dir": str(train_dir),
    }
    save_json(record_path, record)
    started = time.monotonic()
    try:
        print(f"\n{run_name} | {args.epochs} epochs | batch {args.batch} | workers {args.workers}\n", flush=True)
        train_exit = subprocess.run(build_train_command(args, run_name), cwd=ROOT).returncode
        if (train_dir / "results.csv").is_file():
            record["training_result"] = training_metrics(train_dir / "results.csv")
        if train_exit:
            record["status"] = "training_failed"
            record["exit_code"] = train_exit
            return train_exit
        best = train_dir / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"Missing best.pt: {best}")
        record["best_weights"] = str(best)
        if args.no_eval:
            record["status"] = "complete"
            return 0

        record["status"] = "evaluating"
        save_json(record_path, record)
        eval_command = [
            sys.executable, str(ROOT / "src" / "evaluate_small.py"),
            "--weights", str(best), "--split", "val", "--imgsz", str(args.imgsz),
            "--batch", str(args.eval_batch), "--output-dir", str(evaluation_dir), "--summary-only",
        ]
        eval_exit = subprocess.run(eval_command, cwd=ROOT).returncode
        if eval_exit:
            record["status"] = "evaluation_failed"
            record["exit_code"] = eval_exit
            return eval_exit
        full = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
        record["evaluation_result"] = {
            key: full[key] for key in ("AP_small", "AP_medium", "AP_large", "AP", "AP50", "AP75")
        }
        record["evaluation_summary"] = str(evaluation_dir / "summary.json")
        record["status"] = "complete"
        return 0
    except Exception as error:
        record["status"] = "error"
        record["error"] = str(error)
        raise
    finally:
        record["finished_at"] = local_now()
        record["duration_seconds"] = round(time.monotonic() - started, 1)
        save_json(record_path, record)
        print(f"\n{run_name}: {record['status']} | result: {record_path}\n", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
