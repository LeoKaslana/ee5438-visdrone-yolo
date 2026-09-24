"""Resume the interrupted P2/P3/P4 training with its saved settings."""

from pathlib import Path

from ultralytics import YOLO, settings


ROOT = Path(__file__).resolve().parents[1]
LAST_CHECKPOINT = ROOT / "runs" / "p2" / "formal_yolo11n_p2p3p4_640_b16_seed0" / "weights" / "last.pt"


def main() -> None:
    if not LAST_CHECKPOINT.is_file():
        raise SystemExit(f"Resume checkpoint not found: {LAST_CHECKPOINT}")
    settings.update({"datasets_dir": str(ROOT / "data"), "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs")})
    YOLO(str(LAST_CHECKPOINT)).train(resume=True)


if __name__ == "__main__":
    main()
