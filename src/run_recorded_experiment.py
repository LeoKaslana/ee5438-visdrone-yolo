"""Run one VisDrone experiment with immutable run identity and durable logs."""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "VisDrone" / "VisDrone-local.yaml"
PRETRAINED = ROOT / "yolo11n.pt"
MODEL_YAMLS = {
    "baseline": ROOT / "external" / "ultralytics" / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml",
    "p2": ROOT / "configs" / "yolo11n-p2p3p4.yaml",
}
METRIC_KEY = "metrics/mAP50-95(B)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def save_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize_results(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No completed epochs in {path}")
    best = max(rows, key=lambda row: float(row[METRIC_KEY]))
    return {"completed_epochs": len(rows), "last_epoch": rows[-1], "best_builtin_epoch": best}


def run_logged(command: list[str], log_path: Path) -> int:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", errors="replace", buffering=1) as log:
        header = "COMMAND: " + subprocess.list2cmdline(command) + "\n"
        log.write(header)
        print(header, end="", flush=True)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        return process.wait()


def build_training_command(args: argparse.Namespace, run_name: str) -> list[str]:
    script = ROOT / "src" / ("train_baseline.py" if args.variant == "baseline" else "train_p2.py")
    return [
        sys.executable,
        str(script),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--imgsz", str(args.imgsz),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--fraction", str(args.fraction),
        "--save-period", str(args.save_period),
        "--name", run_name,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "p2"), required=True)
    parser.add_argument("--comparison-id", required=True, help="Shared ID for a matched experiment pair")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--no-eval", action="store_true", help="Skip AP-small evaluation, useful for a smoke test")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.comparison_id):
        parser.error("--comparison-id must use only letters, digits, underscores, and hyphens")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.save_period == 0:
        parser.error("Invalid epochs, batch, workers, or save-period")
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    for required in (DATA_YAML, PRETRAINED, MODEL_YAMLS[args.variant]):
        if not required.is_file():
            parser.error(f"Required file not found: {required}")

    run_name = f"{args.comparison_id}-{args.variant}"
    train_dir = ROOT / "runs" / args.variant / run_name
    record_dir = ROOT / "runs" / "records" / run_name
    evaluation_dir = ROOT / "runs" / "evaluation" / f"{run_name}-val"
    if any(path.exists() for path in (train_dir, record_dir, evaluation_dir)):
        parser.error(f"Run identity already exists: {run_name}. Choose a new --comparison-id")
    record_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = record_dir / "manifest.json"

    import torch
    import ultralytics

    manifest = {
        "schema_version": 1,
        "run_name": run_name,
        "comparison_id": args.comparison_id,
        "variant": args.variant,
        "status": "training",
        "started_at_utc": utc_now(),
        "training_config": {
            "epochs": args.epochs, "batch": args.batch, "workers": args.workers,
            "imgsz": args.imgsz, "seed": args.seed, "fraction": args.fraction,
            "save_period": args.save_period, "eval_batch": args.eval_batch,
            "evaluation_enabled": not args.no_eval,
        },
        "paths": {
            "training": str(train_dir), "record": str(record_dir), "evaluation": str(evaluation_dir),
        },
        "code": {"git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--short")},
        "input_sha256": {
            "data_yaml": file_sha256(DATA_YAML),
            "model_yaml": file_sha256(MODEL_YAMLS[args.variant]),
            "pretrained_checkpoint": file_sha256(PRETRAINED),
        },
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data_counts": {
            "train_images": len(list((ROOT / "data" / "VisDrone" / "images" / "train").glob("*.jpg"))),
            "val_images": len(list((ROOT / "data" / "VisDrone" / "images" / "val").glob("*.jpg"))),
        },
    }
    save_manifest(manifest_path, manifest)
    started = time.monotonic()
    try:
        train_command = build_training_command(args, run_name)
        manifest["training_command"] = train_command
        save_manifest(manifest_path, manifest)
        train_exit = run_logged(train_command, record_dir / "train.log")
        manifest["training_exit_code"] = train_exit
        if (train_dir / "results.csv").is_file():
            manifest["training_results"] = summarize_results(train_dir / "results.csv")
        if train_exit:
            manifest["status"] = "training_failed"
            return train_exit
        best = train_dir / "weights" / "best.pt"
        last = train_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise FileNotFoundError(f"Training completed without best.pt or last.pt in {train_dir}")
        manifest["checkpoints"] = {
            "best_sha256": file_sha256(best), "last_sha256": file_sha256(last),
            "periodic": sorted(path.name for path in (train_dir / "weights").glob("epoch*.pt")),
        }
        manifest["status"] = "evaluating" if not args.no_eval else "complete"
        save_manifest(manifest_path, manifest)
        if args.no_eval:
            return 0

        evaluation_command = [
            sys.executable, str(ROOT / "src" / "evaluate_small.py"),
            "--weights", str(best), "--split", "val", "--imgsz", str(args.imgsz),
            "--batch", str(args.eval_batch), "--output-dir", str(evaluation_dir),
        ]
        manifest["evaluation_command"] = evaluation_command
        save_manifest(manifest_path, manifest)
        evaluation_exit = run_logged(evaluation_command, record_dir / "evaluation.log")
        manifest["evaluation_exit_code"] = evaluation_exit
        if evaluation_exit:
            manifest["status"] = "evaluation_failed"
            return evaluation_exit
        manifest["evaluation_results"] = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
        manifest["status"] = "complete"
        return 0
    except Exception as error:
        manifest["status"] = "error"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_at_utc"] = utc_now()
        manifest["wall_seconds"] = round(time.monotonic() - started, 3)
        save_manifest(manifest_path, manifest)
        print(f"\nExperiment {run_name}: {manifest['status']}; record: {manifest_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
