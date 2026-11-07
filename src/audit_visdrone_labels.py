"""Audit original VisDrone ZIP annotations against converted YOLO labels."""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "VisDrone"
NAMES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"
)


def audit_split(split: str) -> dict:
    source_name = "VisDrone2019-DET-test-dev" if split == "test" else f"VisDrone2019-DET-{split}"
    archive = DATA_ROOT / f"{source_name}.zip"
    images = sorted((DATA_ROOT / "images" / split).glob("*.jpg"))
    counts = Counter()
    small_counts = Counter()
    ignored = 0
    annotation_count = 0
    max_coordinate_error = 0.0
    degenerate_counts = Counter()
    with zipfile.ZipFile(archive) as source:
        for image_path in images:
            member = f"{source_name}/annotations/{image_path.stem}.txt"
            original = source.read(member).decode("utf-8").strip().splitlines()
            converted_path = DATA_ROOT / "labels" / split / f"{image_path.stem}.txt"
            converted = converted_path.read_text(encoding="utf-8").strip().splitlines()
            with Image.open(image_path) as image:
                width, height = image.size
            expected = []
            for line in original:
                fields = line.split(",")
                if len(fields) < 6:
                    raise ValueError(f"Malformed source annotation: {member}")
                if int(fields[4]) == 0:
                    ignored += 1
                    continue
                x, y, box_width, box_height = map(int, fields[:4])
                class_id = int(fields[5]) - 1
                if not 0 <= class_id < len(NAMES):
                    raise ValueError(f"Invalid original class in {member}")
                if box_width <= 0 or box_height <= 0:
                    degenerate_counts[NAMES[class_id]] += 1
                expected.append(
                    (class_id, (x + box_width / 2) / width, (y + box_height / 2) / height,
                     box_width / width, box_height / height)
                )
                counts[NAMES[class_id]] += 1
                small_counts[NAMES[class_id]] += box_width * box_height < 32**2
            if len(expected) != len(converted):
                raise ValueError(f"Object count mismatch in {image_path.name}: {len(expected)} vs {len(converted)}")
            for source_box, label in zip(expected, converted):
                values = label.split()
                if len(values) != 5 or int(values[0]) != source_box[0]:
                    raise ValueError(f"Class mismatch in {converted_path}")
                error = max(abs(float(converted_value) - source_value) for converted_value, source_value in zip(values[1:], source_box[1:]))
                max_coordinate_error = max(max_coordinate_error, error)
                if error > 0.00000051:
                    raise ValueError(f"Box mismatch in {converted_path}: {error}")
            annotation_count += len(expected)
    return {
        "images": len(images),
        "objects": annotation_count,
        "ignored_source_regions": ignored,
        "max_normalized_coordinate_error": max_coordinate_error,
        "degenerate_objects_by_class": dict(degenerate_counts),
        "objects_by_class": dict(counts),
        "small_objects_by_class": dict(small_counts),
        "original_and_yolo_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "analysis" / "visdrone_label_audit.json")
    args = parser.parse_args()
    report = {split: audit_split(split) for split in args.splits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
