"""Evaluate VisDrone detections with COCO AP by object size in original pixels."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "VisDrone"
SMALL_AREA = 32**2
MEDIUM_AREA = 96**2


def yolo_box_to_xywh(values: list[float], width: int, height: int) -> tuple[list[float], bool]:
    """Convert normalized center coordinates to a clipped original-pixel COCO box."""
    x_center, y_center, box_width, box_height = values
    if not all(0 <= value <= 1 for value in values) or box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid normalized YOLO box: {values}")

    raw = (
        (x_center - box_width / 2) * width,
        (y_center - box_height / 2) * height,
        (x_center + box_width / 2) * width,
        (y_center + box_height / 2) * height,
    )
    x1, y1, x2, y2 = (
        min(max(raw[0], 0.0), float(width)),
        min(max(raw[1], 0.0), float(height)),
        min(max(raw[2], 0.0), float(width)),
        min(max(raw[3], 0.0), float(height)),
    )
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box has no area after clipping: {values}")
    return [x1, y1, x2 - x1, y2 - y1], any(abs(a - b) > 1e-3 for a, b in zip(raw, (x1, y1, x2, y2)))


def load_names(data_yaml: Path) -> list[str]:
    from ultralytics.utils import YAML

    names = YAML.load(data_yaml)["names"]
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(name) for name in names]


def build_ground_truth(split: str, names: list[str]) -> tuple[dict, dict[str, int], Counter, int]:
    image_dir = DATA_ROOT / "images" / split
    label_dir = DATA_ROOT / "labels" / split
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    dataset = {
        "info": {"description": f"VisDrone {split}, converted YOLO labels, original-pixel areas"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(names)],
    }
    image_ids = {}
    sizes = Counter()
    clipped_count = 0
    annotation_id = 1
    for image_id, image_path in enumerate(images, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label: {label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        image_ids[image_path.name] = image_id
        dataset["images"].append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Expected five YOLO fields at {label_path}:{line_number}")
            class_id = int(fields[0])
            if class_id < 0 or class_id >= len(names):
                raise ValueError(f"Invalid class at {label_path}:{line_number}: {class_id}")
            bbox, clipped = yolo_box_to_xywh([float(value) for value in fields[1:]], width, height)
            clipped_count += int(clipped)
            area = bbox[2] * bbox[3]
            size = "small" if area < SMALL_AREA else "medium" if area < MEDIUM_AREA else "large"
            sizes[size] += 1
            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return dataset, image_ids, sizes, clipped_count


def mean_valid(values: np.ndarray) -> float | None:
    selected = values[values >= 0]
    return float(selected.mean()) if selected.size else None


def per_class_metrics(evaluation: COCOeval, names: list[str], annotations: list[dict]) -> dict:
    precision = evaluation.eval["precision"]  # IoU, recall, class, area, max detections
    area_labels = evaluation.params.areaRngLbl
    small_index = area_labels.index("small")
    all_index = area_labels.index("all")
    ap50_index = int(np.argmin(np.abs(np.asarray(evaluation.params.iouThrs) - 0.5)))
    counts = Counter((annotation["category_id"], annotation["area"] < SMALL_AREA) for annotation in annotations)
    report = {}
    for index, name in enumerate(names):
        category_id = index + 1
        report[name] = {
            "ground_truth": counts[(category_id, False)] + counts[(category_id, True)],
            "small_ground_truth": counts[(category_id, True)],
            "AP": mean_valid(precision[:, :, index, all_index, -1]),
            "AP50": mean_valid(precision[ap50_index, :, index, all_index, -1]),
            "AP_small": mean_valid(precision[:, :, index, small_index, -1]),
        }
    return report


def overall_metrics(evaluation: COCOeval) -> dict[str, float | None]:
    """Read AP at the configured maxDet, avoiding COCOeval.summarize's hardcoded 100."""
    precision = evaluation.eval["precision"]
    areas = evaluation.params.areaRngLbl
    ious = np.asarray(evaluation.params.iouThrs)
    all_index = areas.index("all")
    ap50_index = int(np.argmin(np.abs(ious - 0.5)))
    ap75_index = int(np.argmin(np.abs(ious - 0.75)))
    return {
        "AP": mean_valid(precision[:, :, :, all_index, -1]),
        "AP50": mean_valid(precision[ap50_index, :, :, all_index, -1]),
        "AP75": mean_valid(precision[ap75_index, :, :, all_index, -1]),
        "AP_small": mean_valid(precision[:, :, :, areas.index("small"), -1]),
        "AP_medium": mean_valid(precision[:, :, :, areas.index("medium"), -1]),
        "AP_large": mean_valid(precision[:, :, :, areas.index("large"), -1]),
    }


def detection_coco(coco_gt: COCO, predictions: list[dict]) -> COCO:
    """Construct a COCO detections object, including the valid no-detection case."""
    if predictions:
        return coco_gt.loadRes(predictions)
    coco_dt = COCO()
    coco_dt.dataset = {
        "info": coco_gt.dataset.get("info", {}),
        "licenses": coco_gt.dataset.get("licenses", []),
        "images": coco_gt.dataset["images"],
        "categories": coco_gt.dataset["categories"],
        "annotations": [],
    }
    coco_dt.createIndex()
    return coco_dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=902)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-predictions", action="store_true", help="Reuse the saved prediction JSON")
    parser.add_argument("--summary-only", action="store_true", help="Save only summary.json, not large COCO ground-truth/prediction JSON files")
    args = parser.parse_args()
    if args.summary_only and args.reuse_predictions:
        parser.error("--summary-only cannot be combined with --reuse-predictions")
    if args.max_det < 10:
        parser.error("--max-det must be at least 10 for COCO evaluation")
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")

    data_yaml = DATA_ROOT / "VisDrone-local.yaml"
    if not data_yaml.is_file():
        parser.error(f"Dataset not prepared: {data_yaml}")
    names = load_names(data_yaml)
    ground_truth, image_ids, sizes, clipped_count = build_ground_truth(args.split, names)
    output_dir = args.output_dir or ROOT / "runs" / "evaluation" / f"{args.weights.parent.parent.name}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        (output_dir / "ground_truth_coco.json").write_text(json.dumps(ground_truth), encoding="utf-8")

    prediction_path = output_dir / "predictions_coco.json"
    summary_path = output_dir / "summary.json"
    elapsed = None
    if args.reuse_predictions:
        if not prediction_path.is_file():
            parser.error(f"No cached predictions: {prediction_path}")
        if not summary_path.is_file():
            parser.error(f"Cannot verify cached prediction settings without: {summary_path}")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_settings = {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        }
        if (
            previous["weights"] != str(args.weights.resolve())
            or previous["split"] != args.split
            or previous["settings"] != expected_settings
        ):
            raise RuntimeError("Cached predictions were made with different weights or settings")
        elapsed = previous["prediction_wall_seconds"]
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        if not {prediction["image_id"] for prediction in predictions}.issubset(set(image_ids.values())):
            raise RuntimeError("Cached predictions contain an unknown image; rerun without --reuse-predictions")
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        model_names = [str(model.names[index]) for index in range(len(names))]
        if model_names != names:
            raise ValueError(f"Model classes do not match dataset: {model_names} vs {names}")

        predictions = []
        seen = set()
        started = time.perf_counter()
        results = model.predict(
            source=str(DATA_ROOT / "images" / args.split),
            stream=True,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            quantize=16 if args.device.lower() != "cpu" else None,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            save=False,
        )
        for result in results:
            filename = Path(result.path).name
            if filename not in image_ids or filename in seen:
                raise RuntimeError(f"Unexpected or duplicate prediction image: {filename}")
            seen.add(filename)
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), class_id, score in zip(xyxy, classes, scores):
                predictions.append(
                    {
                        "image_id": image_ids[filename],
                        "category_id": int(class_id) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
        elapsed = time.perf_counter() - started
        if len(seen) != len(image_ids):
            raise RuntimeError(f"Predicted {len(seen)} of {len(image_ids)} images")
        if not args.summary_only:
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_dt = detection_coco(coco_gt, predictions)
    evaluation = COCOeval(coco_gt, coco_dt, "bbox")
    evaluation.params.imgIds = sorted(image_ids.values())
    evaluation.params.catIds = list(range(1, len(names) + 1))
    evaluation.params.maxDets = [1, 10, args.max_det]
    evaluation.evaluate()
    evaluation.accumulate()
    metrics = overall_metrics(evaluation)

    summary = {
        "weights": str(args.weights.resolve()),
        "split": args.split,
        "images": len(image_ids),
        "ground_truth_boxes": len(ground_truth["annotations"]),
        "predicted_boxes": len(predictions),
        "ground_truth_by_size": dict(sizes),
        "small_ground_truth_fraction": sizes["small"] / len(ground_truth["annotations"]),
        "clipped_ground_truth_boxes": clipped_count,
        "size_definition": "COCO original-image pixels: small < 32^2, medium < 96^2, large >= 96^2",
        "evaluation_note": "Uses Ultralytics-converted YOLO labels; not an official VisDrone challenge score.",
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        },
        "prediction_wall_seconds": elapsed,
        **metrics,
        "per_class": per_class_metrics(evaluation, names, ground_truth["annotations"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAP-small: {summary['AP_small']:.4f}; AP: {summary['AP']:.4f}; AP50: {summary['AP50']:.4f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
