"""
Export crops cho Qwen2-VL fine-tuning.

Chạy script này để:
1. Detect plates trong ảnh bằng YOLO
2. Crop và lưu plate crops
3. Tạo manifest CSV cho training

Usage:
    python scripts/export_crops_for_training.py \
        --input-dir data/raw \
        --output-dir data/crops \
        --detector-model weights/yolov8n_plate.pt \
        --manifest data/manifests/train.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.detector.yolov8_detector import YoloV8PlateDetector
from src.io.readers import ImageReader
from src.utils.types import FrameData


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_crops_for_training(
    input_dir: Path,
    output_dir: Path,
    detector_model: Path,
    manifest_path: Path | None = None,
    *,
    crop_margin_ratio: float = 0.1,
    min_crop_size: int = 32,
) -> Path:
    """Export plate crops từ images để train Qwen2-VL.

    Args:
        input_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục lưu crops
        detector_model: Path đến YOLO detector model
        manifest_path: CSV file chứa ground truth (image_id, plate_text)
        crop_margin_ratio: Tỉ lệ margin thêm vào crop
        min_crop_size: Kích thước tối thiểu của crop (pixels)

    Returns:
        Path đến file manifest CSV đã tạo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detector
    logger.info(f"Loading detector: {detector_model}")
    detector = YoloV8PlateDetector(str(detector_model))

    # Load manifest nếu có
    gt_dict: dict[str, str] = {}
    if manifest_path and manifest_path.exists():
        logger.info(f"Loading ground truth from: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_id") or Path(row["image_path"]).stem
                gt_dict[image_id] = row["plate_text"]
        logger.info(f"Loaded {len(gt_dict)} ground truth labels")

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files: list[Path] = []
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f"*{ext}"))

    logger.info(f"Found {len(image_files)} images in {input_dir}")

    # Process images
    records: list[dict] = []
    exported_count = 0
    skipped_count = 0

    for img_path in tqdm(image_files, desc="Exporting crops"):
        try:
            # Read image
            reader = ImageReader()
            frame_data = reader.read_image(img_path)

            # Detect plates
            detections = detector.predict(frame_data)

            if not detections:
                logger.debug(f"No detection in {img_path.name}")
                skipped_count += 1
                continue

            # Get best detection
            best_det = max(detections, key=lambda d: d.score)

            # Calculate crop bbox với margin
            x1, y1, x2, y2 = best_det.bbox_xyxy
            h, w = frame_data.frame.shape[:2]

            # Add margin
            crop_w = x2 - x1
            crop_h = y2 - y1
            margin_x = int(crop_w * crop_margin_ratio)
            margin_y = int(crop_h * crop_margin_ratio)

            x1_crop = max(0, x1 - margin_x)
            y1_crop = max(0, y1 - margin_y)
            x2_crop = min(w, x2 + margin_x)
            y2_crop = min(h, y2 + margin_y)

            # Crop
            crop = frame_data.frame[y1_crop:y2_crop, x1_crop:x2_crop]

            # Check size
            if crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
                logger.debug(f"Crop too small: {img_path.name}")
                skipped_count += 1
                continue

            # Generate output filename
            image_id = img_path.stem
            output_path = output_dir / f"{image_id}.jpg"

            # Save crop
            cv2.imwrite(str(output_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            # Get ground truth
            plate_text = gt_dict.get(image_id, "")

            # Add to records
            records.append({
                "image_id": image_id,
                "image_path": str(output_path),
                "text_gt": plate_text,
            })

            exported_count += 1

        except Exception as e:
            logger.warning(f"Error processing {img_path.name}: {e}")
            skipped_count += 1

    # Save manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "text_gt"])
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Exported {exported_count} crops to {output_dir}")
    logger.info(f"Skipped {skipped_count} images")
    logger.info(f"Manifest saved to: {manifest_path}")

    return manifest_path


def create_dataset_zip(output_dir: Path, zip_path: Path | None = None) -> Path:
    """Tạo ZIP file từ crops để upload lên Colab."""
    import shutil

    if zip_path is None:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"

    shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),
        format="zip",
        root_dir=output_dir,
    )

    logger.info(f"Created ZIP: {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Export plate crops cho Qwen2-VL training"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Thư mục chứa ảnh gốc",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Thư mục lưu crops",
    )
    parser.add_argument(
        "--detector-model",
        type=Path,
        required=True,
        help="Path đến YOLO detector model",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV file chứa ground truth (image_id, plate_text)",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.1,
        help="Tỉ lệ margin thêm vào crop (default: 0.1)",
    )
    parser.add_argument(
        "--create-zip",
        action="store_true",
        help="Tạo ZIP file sau khi export",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Kích thước tối thiểu của crop (default: 32)",
    )

    args = parser.parse_args()

    # Export crops
    manifest_path = export_crops_for_training(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        detector_model=args.detector_model,
        manifest_path=args.manifest,
        crop_margin_ratio=args.margin_ratio,
        min_crop_size=args.min_size,
    )

    # Create ZIP if requested
    if args.create_zip:
        create_dataset_zip(args.output_dir)


if __name__ == "__main__":
    main()
