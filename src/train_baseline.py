"""Train the unchanged YOLO11n baseline on the local VisDrone dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO, settings


ROOT = Path(__file__).resolve().parents[1]
settings.update({"datasets_dir": str(ROOT / "data"), "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs")})
DATA_YAML = ROOT / "data" / "VisDrone" / "VisDrone-local.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fraction", type=float, default=1.0)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
if not DATA_YAML.exists():
    raise SystemExit("VisDrone is not prepared. Run: python src/prepare_visdrone.py")
run_name = f"yolo11n_{args.imgsz}_seed{args.seed}"
if args.fraction < 1.0:
    run_name = f"pilot_{run_name}_fraction{args.fraction:g}"

model = YOLO("yolo11n.pt")
model.train(
    data=str(DATA_YAML),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    seed=args.seed,
    device=0,
    workers=args.workers,
    amp=True,
    cache=False,
    fraction=args.fraction,
    project=str(ROOT / "runs" / "baseline"),
    name=run_name,
)
