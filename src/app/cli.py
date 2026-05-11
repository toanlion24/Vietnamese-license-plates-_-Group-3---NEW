from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json
from itertools import islice

from src.detector.base import DummyCenterDetector
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.io.readers import iter_images
from src.ocr.base import DummyOcr
from src.ocr.easyocr_adapter import EasyOcrAdapter
from src.ocr.ensemble_easy_trocr import EasyTrocrEnsembleOcr
from src.ocr.trocr_adapter import TrOcrAdapter
from src.pipeline.infer_plate_pipeline import PlateInferencePipeline


def _build_detector(args: argparse.Namespace):
    if args.detector_backend == "dummy":
        return DummyCenterDetector()
    return YoloV8PlateDetector(
        model_path=args.detector_model,
        conf_threshold=args.detector_conf,
        iou=float(args.yolo_iou),
        imgsz=int(args.yolo_imgsz),
    )


def _build_ocr(args: argparse.Namespace):
    if args.ocr_backend == "dummy":
        return DummyOcr()
    if args.ocr_backend == "easyocr":
        return EasyOcrAdapter(gpu=args.ocr_gpu)
    if args.ocr_backend == "ensemble_easy_trocr":
        easy = EasyOcrAdapter(gpu=args.ocr_gpu)
        trocr = TrOcrAdapter(
            model_name=args.trocr_model,
            device=args.device,
            cache_dir=str(args.model_cache_dir) if args.model_cache_dir else None,
        )
        return EasyTrocrEnsembleOcr(easy, trocr, aggressive_post=bool(args.aggressive_postprocess))
    return TrOcrAdapter(
        model_name=args.trocr_model,
        device=args.device,
        cache_dir=str(args.model_cache_dir) if args.model_cache_dir else None,
    )


def run_batch(args: argparse.Namespace) -> None:
    pipeline = PlateInferencePipeline(
        detector=_build_detector(args),
        ocr=_build_ocr(args),
        crop_margin_ratio=float(args.crop_margin_ratio),
        preprocess_clahe=bool(args.preprocess_clahe),
        aggressive_postprocess=bool(args.aggressive_postprocess),
    )
    rows: list[dict[str, object]] = []
    frames = iter_images(args.input_dir)
    if args.max_images is not None:
        frames = islice(frames, args.max_images)

    for frame in frames:
        result = pipeline.run(frame)
        rows.append(
            {
                "image_id": result.image_id,
                "plate_text": result.plate_text,
                "bbox_xyxy": result.bbox_xyxy,
                "confidence": result.confidence,
                "latency_ms": result.timestamp_ms,
                "source": result.source,
            }
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN license plate inference on image folder.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument("--detector-backend", choices=["yolov8", "dummy"], default="yolov8")
    parser.add_argument("--detector-model", type=Path, default=Path("weights/yolov8_license_plate.pt"))
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.0)
    parser.add_argument("--preprocess-clahe", action="store_true")
    parser.add_argument("--aggressive-postprocess", action="store_true")
    parser.add_argument(
        "--ocr-backend",
        choices=["easyocr", "trocr", "dummy", "ensemble_easy_trocr"],
        default="easyocr",
    )
    parser.add_argument("--ocr-gpu", action="store_true")
    parser.add_argument("--trocr-model", type=str, default="microsoft/trocr-base-printed")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for quick demos/debug runs.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    run_batch(args)

