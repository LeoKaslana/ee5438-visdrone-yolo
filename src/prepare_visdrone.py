"""Download the official VisDrone split through Ultralytics and check labels."""

from pathlib import Path

from ultralytics import settings
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "VisDrone"
settings.update({"datasets_dir": str(ROOT / "data"), "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs")})

from ultralytics.data.utils import check_det_dataset  # noqa: E402


official_yaml = ROOT / "external" / "ultralytics" / "ultralytics" / "cfg" / "datasets" / "VisDrone.yaml"
source = YAML.load(official_yaml)
if not all((DATA_ROOT / "images" / split).exists() for split in ("train", "val", "test")):
    source["path"] = DATA_ROOT
    exec(source["download"], {"yaml": source})

local_yaml = DATA_ROOT / "VisDrone-local.yaml"
YAML.save(local_yaml, {key: source[key] for key in ("train", "val", "test", "names")} | {"path": str(DATA_ROOT)})
data = check_det_dataset(str(local_yaml), autodownload=False)
expected = {"train": 6471, "val": 548, "test": 1610}
for split, count in expected.items():
    images = list((DATA_ROOT / "images" / split).glob("*.jpg"))
    labels = list((DATA_ROOT / "labels" / split).glob("*.txt"))
    print(f"{split}: {len(images)} images, {len(labels)} labels (expected {count})")
    if len(images) != count or len(labels) != count:
        raise RuntimeError(f"Incomplete {split} split")
    for label in labels[:20]:
        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                raise RuntimeError(f"Malformed YOLO label: {label}")
            class_id = int(values[0])
            box = [float(x) for x in values[1:]]
            if not 0 <= class_id < 10 or not all(0 <= x <= 1 for x in box):
                raise RuntimeError(f"Out-of-range YOLO label: {label}")

print(f"Dataset ready: {data['path']}")
