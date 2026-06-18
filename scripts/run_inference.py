"""
Inference script cho VN License Plate Recognition với Qwen2-VL.

Chạy inference trên ảnh, video, hoặc webcam sử dụng:
- YOLOv8n cho plate detection
- Qwen2-VL cho OCR (từ Hugging Face)

Usage:
    # Single image
    python scripts/run_inference.py --image data/test.jpg --output result.json

    # Batch from folder
    python scripts/run_inference.py --input-dir data/test_images --output-dir outputs/

    # Video
    python scripts/run_inference.py --video data/test.mp4 --output video_results.json

    # Webcam
    python scripts/run_inference.py --webcam --output webcam_results.json

    # Load từ Hugging Face
    python scripts/run_inference.py --image data/test.jpg --qwen-model toannv1990/vn-plate-qwen2-vl-2b
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from src.detector.yolov8_detector import YoloV8PlateDetector
from src.io.readers import ImageReader, VideoReader
from src.ocr.qwen_adapter import Qwen2VLPlateOcr
from src.pipeline.infer_plate_pipeline import PlateInferencePipeline
from src.postprocess.plate_rules import (
    advanced_repair_ocr_text,
    is_valid_vn_plate,
    normalize_plate_text,
)
from src.utils.types import FrameData, PipelineResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models(
    detector_model: str,
    qwen_model: str,
    device: str = "auto",
) -> tuple[YoloV8PlateDetector, Qwen2VLPlateOcr]:
    """Load detector và OCR models."""
    logger.info(f"Loading detector: {detector_model}")
    detector = YoloV8PlateDetector(detector_model)

    logger.info(f"Loading Qwen2-VL: {qwen_model}")
    ocr = Qwen2VLPlateOcr(
        model_name=qwen_model,
        device=device,
    )

    return detector, ocr


def recognize_single_image(
    image_path: Path,
    detector: YoloV8PlateDetector,
    ocr: Qwen2VLPlateOcr,
    *,
    crop_margin_ratio: float = 0.1,
    preprocess_clahe: bool = False,
) -> dict:
    """Nhận diện biển số từ một ảnh."""
    start_time = time.perf_counter()

    # Read image
    reader = ImageReader()
    frame_data = reader.read_image(image_path)

    # Create pipeline
    pipeline = PlateInferencePipeline(
        detector=detector,
        ocr=ocr,
        crop_margin_ratio=crop_margin_ratio,
        preprocess_clahe=preprocess_clahe,
    )

    # Run inference
    result = pipeline.run(frame_data)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "image_path": str(image_path),
        "plate_text": result.plate_text,
        "confidence": float(result.confidence),
        "bbox": result.bbox_xyxy,
        "latency_ms": elapsed_ms,
    }


def recognize_batch(
    input_dir: Path,
    output_dir: Path,
    detector: YoloV8PlateDetector,
    ocr: Qwen2VLPlateOcr,
    *,
    crop_margin_ratio: float = 0.1,
    preprocess_clahe: bool = False,
    save_visualizations: bool = False,
) -> list[dict]:
    """Nhận diện biển số từ batch ảnh."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files: list[Path] = []
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f"*{ext}"))

    logger.info(f"Found {len(image_files)} images")

    results = []
    reader = ImageReader()
    pipeline = PlateInferencePipeline(
        detector=detector,
        ocr=ocr,
        crop_margin_ratio=crop_margin_ratio,
        preprocess_clahe=preprocess_clahe,
    )

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            frame_data = reader.read_image(img_path)
            result = pipeline.run(frame_data)

            results.append({
                "image_path": str(img_path),
                "plate_text": result.plate_text,
                "confidence": float(result.confidence),
                "bbox": result.bbox_xyxy,
            })

            # Save visualization
            if save_visualizations:
                _save_visualization(
                    img_path,
                    output_dir / f"{img_path.stem}_result.jpg",
                    result,
                )

        except Exception as e:
            logger.warning(f"Error processing {img_path.name}: {e}")
            results.append({
                "image_path": str(img_path),
                "plate_text": "",
                "confidence": 0.0,
                "bbox": None,
                "error": str(e),
            })

    return results


def recognize_video(
    video_path: Path,
    output_path: Path,
    detector: YoloV8PlateDetector,
    ocr: Qwen2VLPlateOcr,
    *,
    crop_margin_ratio: float = 0.1,
    preprocess_clahe: bool = False,
    fps: int | None = None,
) -> list[dict]:
    """Nhận diện biển số từ video."""
    reader = VideoReader(video_path)
    pipeline = PlateInferencePipeline(
        detector=detector,
        ocr=ocr,
        crop_margin_ratio=crop_margin_ratio,
        preprocess_clahe=preprocess_clahe,
    )

    # Video writer
    cap = cv2.VideoCapture(str(video_path))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps or int(fps_in)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )

    results = []
    frame_idx = 0

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Input FPS: {fps_in}, Output FPS: {fps_out}")

    while True:
        frame_data = reader.read_frame()
        if frame_data is None:
            break

        result = pipeline.run(frame_data)
        results.append({
            "frame_idx": frame_idx,
            "timestamp_ms": frame_data.timestamp_ms,
            "plate_text": result.plate_text,
            "confidence": float(result.confidence),
            "bbox": result.bbox_xyxy,
        })

        # Draw on frame
        frame = _draw_result(frame_data.frame, result)
        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    logger.info(f"Processed {frame_idx} frames")
    return results


def recognize_webcam(
    detector: YoloV8PlateDetector,
    ocr: Qwen2VLPlateOcr,
    *,
    crop_margin_ratio: float = 0.1,
    preprocess_clahe: bool = False,
    camera_idx: int = 0,
) -> None:
    """Nhận diện biển số realtime từ webcam."""
    pipeline = PlateInferencePipeline(
        detector=detector,
        ocr=ocr,
        crop_margin_ratio=crop_margin_ratio,
        preprocess_clahe=preprocess_clahe,
    )

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {camera_idx}")

    logger.info("Press 'q' to quit")
    logger.info("Press 's' to save screenshot")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = FrameData(
            image_id=f"webcam_{frame_idx}",
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            source="webcam",
            timestamp_ms=frame_idx * 33.33,
        )

        result = pipeline.run(frame_data)

        # Draw
        display_frame = _draw_result(frame, result)
        cv2.imshow("VN Plate Recognition (Press 'q' to quit)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"screenshot_{frame_idx}.jpg", frame)
            logger.info(f"Saved screenshot_{frame_idx}.jpg")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def _draw_result(frame: np.ndarray, result: PipelineResult) -> np.ndarray:
    """Vẽ kết quả lên frame."""
    # Convert to BGR for OpenCV
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        display = frame.copy()

    # Draw bbox
    if result.bbox_xyxy is not None:
        x1, y1, x2, y2 = result.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw text
    text = f"{result.plate_text} ({result.confidence:.2f})"
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return display


def _save_visualization(
    input_path: Path,
    output_path: Path,
    result: PipelineResult,
) -> None:
    """Lưu ảnh có vẽ kết quả."""
    frame = cv2.imread(str(input_path))
    if frame is None:
        return

    display = _draw_result(frame, result)
    cv2.imwrite(str(output_path), display)


def save_results(results: list[dict], output_path: Path) -> None:
    """Lưu kết quả ra file JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")


def print_summary(results: list[dict]) -> None:
    """In tóm tắt kết quả."""
    total = len(results)
    valid_plates = sum(1 for r in results if is_valid_vn_plate(r.get("plate_text", "")))
    avg_confidence = np.mean([r.get("confidence", 0) for r in results])
    avg_latency = np.mean([r.get("latency_ms", 0) for r in results])

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total images: {total}")
    print(f"Valid plates: {valid_plates} ({valid_plates / total * 100:.1f}%)")
    print(f"Avg confidence: {avg_confidence:.3f}")
    if avg_latency > 0:
        print(f"Avg latency: {avg_latency:.1f}ms")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="VN License Plate Recognition với YOLOv8 + Qwen2-VL"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Single image file")
    input_group.add_argument("--input-dir", type=Path, help="Input directory")
    input_group.add_argument("--video", type=Path, help="Video file")
    input_group.add_argument("--webcam", action="store_true", help="Use webcam")

    # Output options
    parser.add_argument("--output", type=Path, help="Output JSON file (single image)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (batch)")
    parser.add_argument("--output-video", type=Path, help="Output video file (video input)")

    # Model options
    parser.add_argument(
        "--detector-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO detector model (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Qwen2-VL model from Hugging Face (default: base model)",
    )

    # Processing options
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.1,
        help="Crop margin ratio (default: 0.1)",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Use CLAHE preprocessing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--camera-idx",
        type=int,
        default=0,
        help="Webcam camera index (default: 0)",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualizations",
    )

    args = parser.parse_args()

    # Load models
    detector, ocr = load_models(args.detector_model, args.qwen_model, args.device)

    # Run inference
    if args.image:
        result = recognize_single_image(
            args.image, detector, ocr,
            crop_margin_ratio=args.crop_margin,
            preprocess_clahe=args.clahe,
        )
        print(f"\nImage: {args.image}")
        print(f"Plate: {result['plate_text']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Latency: {result['latency_ms']:.1f}ms")

        if args.output:
            save_results([result], args.output)

    elif args.input_dir:
        results = recognize_batch(
            args.input_dir, args.output_dir or Path("outputs"),
            detector, ocr,
            crop_margin_ratio=args.crop_margin,
            preprocess_clahe=args.clahe,
            save_visualizations=args.save_viz,
        )
        print_summary(results)

        if args.output:
            save_results(results, args.output)

    elif args.video:
        results = recognize_video(
            args.video, args.output_video or Path("output_video.mp4"),
            detector, ocr,
            crop_margin_ratio=args.crop_margin,
            preprocess_clahe=args.clahe,
        )
        print_summary(results)

        if args.output:
            save_results(results, args.output)

    elif args.webcam:
        recognize_webcam(
            detector, ocr,
            crop_margin_ratio=args.crop_margin,
            preprocess_clahe=args.clahe,
            camera_idx=args.camera_idx,
        )


if __name__ == "__main__":
    main()
