"""Train the YOLO11n P2/P3/P4 variant on the local VisDrone dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO, settings


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "VisDrone" / "VisDrone-local.yaml"
MODEL_YAML = ROOT / "configs" / "yolo11n-p2p3p4.yaml"
PRETRAINED = ROOT / "yolo11n.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-period", type=int, default=-1, help="Save a checkpoint every N epochs; -1 disables it")
    parser.add_argument("--name", type=str, default="formal_yolo11n_p2p3p4_640_b16_seed0")
    args = parser.parse_args()

    for required in (DATA_YAML, MODEL_YAML, PRETRAINED):
        if not required.is_file():
            parser.error(f"Required file not found: {required}")
    settings.update({"datasets_dir": str(ROOT / "data"), "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs")})
    model = YOLO(str(MODEL_YAML)).load(str(PRETRAINED))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=0,
        workers=args.workers,
        save_period=args.save_period,
        amp=True,
        cache=False,
        fraction=args.fraction,
        project=str(ROOT / "runs" / "p2"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
