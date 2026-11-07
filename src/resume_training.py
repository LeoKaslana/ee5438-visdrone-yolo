"""Resume any project YOLO training run from its last optimizer checkpoint."""

import argparse
from pathlib import Path

from ultralytics import YOLO, settings


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Resume checkpoint not found: {args.checkpoint}")
    settings.update({"datasets_dir": str(ROOT / "data"), "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs")})
    YOLO(str(args.checkpoint)).train(resume=True, workers=args.workers)


if __name__ == "__main__":
    main()
