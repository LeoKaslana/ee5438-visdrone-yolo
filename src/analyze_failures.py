"""Compare three saved COCO prediction files and render representative VisDrone errors."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("baseline", "p2", "p2p3p4p5")
SMALL_AREA = 32**2
LARGE_AREA = 96**2


def iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    overlap = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1)) * max(
        0, min(ay1 + ah, by1 + bh) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - overlap
    return overlap / union if union else 0.0


def match_image(annotations: list[dict], predictions: list[dict], threshold: float) -> set[int]:
    """Greedy one-to-one, class-aware IoU=.5 matches at a display confidence threshold."""
    matched = set()
    for prediction in sorted(
        (item for item in predictions if item["score"] >= threshold),
        key=lambda item: item["score"],
        reverse=True,
    ):
        candidates = (
            (iou_xywh(prediction["bbox"], annotation["bbox"]), annotation["id"])
            for annotation in annotations
            if annotation["category_id"] == prediction["category_id"] and annotation["id"] not in matched
        )
        overlap, annotation_id = max(candidates, default=(0.0, None))
        if overlap >= 0.5:
            matched.add(annotation_id)
    return matched


def area_group(box: list[float]) -> str:
    area = box[2] * box[3]
    return "small" if area < SMALL_AREA else "large" if area >= LARGE_AREA else "medium"


def draw_crop(
    image: Image.Image,
    box: list[float],
    annotations: list[dict],
    predictions: list[dict],
    category_id: int,
    title: str,
    threshold: float,
    names: dict[int, str],
) -> Image.Image:
    width, height = image.size
    x, y, bw, bh = box
    crop_width, crop_height = min(width, 512), min(height, 384)
    left = max(0, min(round(x + bw / 2 - crop_width / 2), width - crop_width))
    top = max(0, min(round(y + bh / 2 - crop_height / 2), height - crop_height))
    canvas = Image.new("RGB", (crop_width, crop_height + 28), "#111827")
    canvas.paste(image.crop((left, top, left + crop_width, top + crop_height)), (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill="white")
    for annotation in annotations:
        if annotation["category_id"] != category_id:
            continue
        ax, ay, aw, ah = annotation["bbox"]
        if ax + aw < left or ax > left + crop_width or ay + ah < top or ay > top + crop_height:
            continue
        draw.rectangle((ax - left, ay - top + 28, ax + aw - left, ay + ah - top + 28), outline="#00ff80", width=2)
    for prediction in predictions:
        if prediction["score"] < threshold:
            continue
        is_other_class = prediction["category_id"] != category_id
        if is_other_class and iou_xywh(prediction["bbox"], box) < 0.5:
            continue
        px, py, pw, ph = prediction["bbox"]
        if px + pw < left or px > left + crop_width or py + ph < top or py > top + crop_height:
            continue
        color = "#00d9ff" if is_other_class else "#ffcc00"
        draw.rectangle((px - left, py - top + 28, px + pw - left, py + ph - top + 28), outline=color, width=2)
        if is_other_class:
            draw.text((px - left, max(28, py - top + 16)), names[prediction["category_id"]], fill=color)
    draw.rectangle((x - left, y - top + 28, x + bw - left, y + bh - top + 28), outline="#ff00e6", width=3)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-id", default="strict-b8w0-s0-20260925-011512")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    output = ROOT / "runs" / "analysis" / args.comparison_id
    output.mkdir(parents=True, exist_ok=True)

    directories = {
        variant: ROOT / "runs" / "analysis" / f"{args.comparison_id}-{variant}" for variant in VARIANTS
    }
    ground_truth = json.loads((directories["baseline"] / "ground_truth_coco.json").read_text(encoding="utf-8"))
    names = {category["id"]: category["name"] for category in ground_truth["categories"]}
    images = {image["id"]: image for image in ground_truth["images"]}
    annotations_by_image = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    predictions_by_variant = {}
    confident_prediction_counts = {}
    for variant, directory in directories.items():
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["settings"]["conf"] != 0.001 or summary["settings"]["iou"] != 0.7:
            raise ValueError(f"Different prediction settings: {variant}")
        by_image = defaultdict(list)
        confident_prediction_counts[variant] = Counter()
        for prediction in json.loads((directory / "predictions_coco.json").read_text(encoding="utf-8")):
            by_image[prediction["image_id"]].append(prediction)
            if prediction["score"] >= args.conf:
                confident_prediction_counts[variant][names[prediction["category_id"]]] += 1
        predictions_by_variant[variant] = by_image

    matched = {variant: set() for variant in VARIANTS}
    for image_id, annotations in annotations_by_image.items():
        for variant in VARIANTS:
            matched[variant].update(match_image(annotations, predictions_by_variant[variant][image_id], args.conf))

    counts = {variant: Counter() for variant in VARIANTS}
    total = Counter()
    candidates = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        category = names[annotation["category_id"]]
        size = area_group(annotation["bbox"])
        key = (category, size)
        total[key] += 1
        for variant in VARIANTS:
            counts[variant][key] += annotation["id"] in matched[variant]
        baseline, p2, four = (annotation["id"] in matched[variant] for variant in VARIANTS)
        if size == "small" and category in ("pedestrian", "car", "motor") and four and not baseline:
            candidates[f"{category}_four_gain"].append(annotation)
        if size == "small" and category in ("bicycle", "awning-tricycle") and baseline and not four:
            candidates[f"{category}_four_loss"].append(annotation)
        if size == "large" and four and not p2:
            candidates["large_four_restored"].append(annotation)
        if size == "small" and not four and category in ("bicycle", "awning-tricycle"):
            target = "motor" if category == "bicycle" else "car"
            if any(
                prediction["score"] >= args.conf
                and names[prediction["category_id"]] == target
                and iou_xywh(prediction["bbox"], annotation["bbox"]) >= 0.5
                for prediction in predictions_by_variant["p2p3p4p5"][annotation["image_id"]]
            ):
                candidates[f"{category}_misclassified_as_{target}"].append(annotation)

    selected = {}
    for label, items in candidates.items():
        # Prefer visibly sized objects without selecting a medium/large object for a small-object case.
        annotation = max(items, key=lambda item: item["bbox"][2] * item["bbox"][3])
        image_id = annotation["image_id"]
        image_path = ROOT / "data" / "VisDrone" / "images" / "val" / images[image_id]["file_name"]
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            panels = [
                draw_crop(rgb, annotation["bbox"], annotations_by_image[image_id], [], annotation["category_id"], "Ground truth", args.conf, names)
            ]
            for variant in VARIANTS:
                panels.append(
                    draw_crop(
                        rgb,
                        annotation["bbox"],
                        annotations_by_image[image_id],
                        predictions_by_variant[variant][image_id],
                        annotation["category_id"],
                        variant,
                        args.conf,
                        names,
                    )
                )
            montage = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)))
            cursor = 0
            for panel in panels:
                montage.paste(panel, (cursor, 0))
                cursor += panel.width
            path = output / f"{label}.png"
            montage.save(path)
        selected[label] = {
            "image": images[image_id]["file_name"],
            "annotation_id": annotation["id"],
            "class": names[annotation["category_id"]],
            "size": area_group(annotation["bbox"]),
            "matched": {variant: annotation["id"] in matched[variant] for variant in VARIANTS},
            "montage": str(path),
        }

    overlapping_error_labels = {}
    for variant in VARIANTS:
        overlapping_error_labels[variant] = {}
        for category in ("bicycle", "awning-tricycle"):
            errors = Counter()
            for annotation in ground_truth["annotations"]:
                if (
                    names[annotation["category_id"]] != category
                    or area_group(annotation["bbox"]) != "small"
                    or annotation["id"] in matched[variant]
                ):
                    continue
                nearby = (
                    prediction
                    for prediction in predictions_by_variant[variant][annotation["image_id"]]
                    if prediction["score"] >= args.conf
                    and iou_xywh(prediction["bbox"], annotation["bbox"]) >= 0.5
                )
                best = max(nearby, key=lambda item: item["score"], default=None)
                errors[names[best["category_id"]] if best else "no_overlapping_prediction"] += 1
            overlapping_error_labels[variant][category] = dict(errors)

    confidence_sweep = {}
    for variant in VARIANTS:
        confidence_sweep[variant] = {}
        for category in ("bicycle", "awning-tricycle"):
            category_id = next(index for index, name in names.items() if name == category)
            by_threshold = {}
            for threshold in (0.001, 0.01, 0.05, 0.1, 0.25):
                hits = 0
                detections = 0
                for image_id, annotations in annotations_by_image.items():
                    class_annotations = [item for item in annotations if item["category_id"] == category_id]
                    class_predictions = [
                        item for item in predictions_by_variant[variant][image_id]
                        if item["category_id"] == category_id and item["score"] >= threshold
                    ]
                    detections += len(class_predictions)
                    image_hits = match_image(class_annotations, class_predictions, threshold)
                    hits += sum(
                        item["id"] in image_hits and area_group(item["bbox"]) == "small"
                        for item in class_annotations
                    )
                by_threshold[str(threshold)] = {"small_gt_matched": hits, "predictions_all_sizes": detections}
            confidence_sweep[variant][category] = by_threshold

    report = {
        "comparison_id": args.comparison_id,
        "matching_rule": f"Greedy class-aware IoU >= 0.5, detection confidence >= {args.conf}; diagnostic recall only, not AP",
        "box_legend": "green = all ground truth of selected class, yellow = same-class predictions, cyan = overlapping other-class predictions, magenta = selected ground truth",
        "ground_truth_counts": {f"{category}/{size}": count for (category, size), count in total.items()},
        "matched_counts": {
            variant: {f"{category}/{size}": count for (category, size), count in value.items()}
            for variant, value in counts.items()
        },
        "confident_prediction_counts": {variant: dict(counts) for variant, counts in confident_prediction_counts.items()},
        "unmatched_small_object_overlapping_prediction": overlapping_error_labels,
        "weak_class_confidence_sweep": confidence_sweep,
        "selected_cases": selected,
    }
    path = output / "failure_analysis.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(selected)} case montages and diagnostic counts to {output}")


if __name__ == "__main__":
    main()
