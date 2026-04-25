from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick EDA for YOLO license plate dataset (images + labels)."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    parser.add_argument("--num-preview", type=int, default=12)
    return parser.parse_args()


def load_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        boxes.append((cls_id, xc, yc, w, h))
    return boxes


def label_path_candidates_for_image(
    image_path: Path,
    images_dir: Path,
    labels_dir: Path,
) -> list[Path]:
    relative_image = image_path.relative_to(images_dir)
    rel_txt = relative_image.with_suffix(".txt")
    candidates: list[Path] = [labels_dir / rel_txt]

    # Common mismatch: images in data/images/raw/* but labels in data/labels/raw/*
    if len(relative_image.parts) > 1:
        candidates.append(labels_dir / Path(*relative_image.parts[1:]).with_suffix(".txt"))

    # Also try by filename only for flat label folders.
    candidates.append(labels_dir / f"{image_path.stem}.txt")
    return candidates


def resolve_label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    candidates = label_path_candidates_for_image(image_path, images_dir, labels_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def yolo_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x1 = int((xc - w / 2.0) * img_w)
    y1 = int((yc - h / 2.0) * img_h)
    x2 = int((xc + w / 2.0) * img_w)
    y2 = int((yc + h / 2.0) * img_h)
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def brightness_tag(gray) -> str:
    mean_val = float(gray.mean())
    if mean_val < 70:
        return "dark"
    if mean_val > 170:
        return "bright"
    return "normal"


def draw_preview(
    image_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    output_path: Path,
    cv2,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for cls_id, xc, yc, bw, bh in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"cls:{cls_id}",
            (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def main() -> None:
    args = parse_args()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is not installed. Run: pip install -r requirements.txt"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in args.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise RuntimeError(f"No images found in {args.images_dir}")

    total_images = 0
    images_with_labels = 0
    total_boxes = 0
    widths: list[int] = []
    heights: list[int] = []
    plate_area_ratios: list[float] = []
    brightness_counts = {"dark": 0, "normal": 0, "bright": 0}
    per_image_box_counts: list[int] = []
    missing_label_files: list[str] = []

    preview_dir = args.output_dir / "preview_samples"

    for idx, image_path in enumerate(images):
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        total_images += 1
        img_h, img_w = img.shape[:2]
        widths.append(img_w)
        heights.append(img_h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_counts[brightness_tag(gray)] += 1

        label_path = resolve_label_path_for_image(image_path, args.images_dir, args.labels_dir)
        boxes = load_label_file(label_path)
        if not label_path.exists():
            missing_label_files.append(str(label_path))

        box_count = len(boxes)
        per_image_box_counts.append(box_count)
        if box_count > 0:
            images_with_labels += 1
            total_boxes += box_count
            for _, _, _, bw, bh in boxes:
                plate_area_ratios.append(float(bw * bh))

        if idx < args.num_preview:
            preview_name = image_path.relative_to(args.images_dir).with_suffix("").as_posix().replace("/", "__")
            draw_preview(image_path, boxes, preview_dir / f"{preview_name}_preview.jpg", cv2)

    def mean(values: list[float] | list[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    report = {
        "dataset_overview": {
            "images_total": total_images,
            "images_with_labels": images_with_labels,
            "images_without_labels": total_images - images_with_labels,
            "total_boxes": total_boxes,
            "avg_boxes_per_image": mean(per_image_box_counts),
        },
        "image_size_stats": {
            "avg_width": mean(widths),
            "avg_height": mean(heights),
            "min_width": min(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_width": max(widths) if widths else 0,
            "max_height": max(heights) if heights else 0,
        },
        "plate_box_stats": {
            "avg_area_ratio": mean(plate_area_ratios),
            "min_area_ratio": min(plate_area_ratios) if plate_area_ratios else 0.0,
            "max_area_ratio": max(plate_area_ratios) if plate_area_ratios else 0.0,
        },
        "brightness_distribution": brightness_counts,
        "preview_samples_dir": str(preview_dir),
        "missing_label_files_preview": missing_label_files[:20],
    }

    report_json = args.output_dir / "dataset_report.json"
    report_md = args.output_dir / "dataset_report.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Dataset EDA Report",
        "",
        "## Overview",
        f"- images_total: {report['dataset_overview']['images_total']}",
        f"- images_with_labels: {report['dataset_overview']['images_with_labels']}",
        f"- images_without_labels: {report['dataset_overview']['images_without_labels']}",
        f"- total_boxes: {report['dataset_overview']['total_boxes']}",
        f"- avg_boxes_per_image: {report['dataset_overview']['avg_boxes_per_image']:.3f}",
        "",
        "## Image Size Stats",
        f"- avg_width: {report['image_size_stats']['avg_width']:.2f}",
        f"- avg_height: {report['image_size_stats']['avg_height']:.2f}",
        f"- min_width: {report['image_size_stats']['min_width']}",
        f"- min_height: {report['image_size_stats']['min_height']}",
        f"- max_width: {report['image_size_stats']['max_width']}",
        f"- max_height: {report['image_size_stats']['max_height']}",
        "",
        "## Plate Box Stats",
        f"- avg_area_ratio: {report['plate_box_stats']['avg_area_ratio']:.6f}",
        f"- min_area_ratio: {report['plate_box_stats']['min_area_ratio']:.6f}",
        f"- max_area_ratio: {report['plate_box_stats']['max_area_ratio']:.6f}",
        "",
        "## Brightness Distribution",
        f"- dark: {report['brightness_distribution']['dark']}",
        f"- normal: {report['brightness_distribution']['normal']}",
        f"- bright: {report['brightness_distribution']['bright']}",
        "",
        "## Outputs",
        f"- JSON report: `{report_json}`",
        f"- Preview images with boxes: `{preview_dir}`",
        "",
    ]
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"EDA completed: {report_json}")
    print(f"Preview images: {preview_dir}")


if __name__ == "__main__":
    main()
