from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.base import DummyCenterDetector
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.ocr.base import DummyOcr
from src.ocr.easyocr_adapter import EasyOcrAdapter
from src.ocr.ensemble_easy_trocr import EasyTrocrEnsembleOcr
from src.ocr.trocr_adapter import TrOcrAdapter
from src.pipeline.detailed_plate_infer import infer_plate_detailed
from src.utils.types import FrameData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN plate inference on a video file.")
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/video_predictions.json"))
    parser.add_argument("--output-video", type=Path, default=None, help="Optional annotated output video path.")
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
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame (>=1).")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


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


def _infer_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "crop_margin_ratio": float(args.crop_margin_ratio),
        "preprocess_clahe": bool(args.preprocess_clahe),
        "aggressive_postprocess": bool(args.aggressive_postprocess),
    }


def _draw_result(frame, text: str, bbox_xyxy: tuple[int, int, int, int] | None) -> None:
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        y_text = max(y1 - 8, 15)
    else:
        y_text = 25
    cv2.putText(frame, text or "<no_plate>", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    if not args.input_video.is_file():
        raise SystemExit(f"Video not found: {args.input_video}")

    detector = _build_detector(args)
    ocr = _build_ocr(args)
    infer_kw = _infer_kwargs(args)

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if args.output_video is not None:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output_video), fourcc, fps if fps > 0 else 25.0, (width, height))

    rows: list[dict[str, object]] = []
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.frame_stride != 0:
            frame_idx += 1
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        image_id = f"{args.input_video.stem}_f{frame_idx:06d}"
        frame_data = FrameData(image_id=image_id, frame=frame, source=str(args.input_video), timestamp_ms=timestamp_ms)
        result = infer_plate_detailed(detector, ocr, frame_data, **infer_kw)
        rows.append(
            {
                "image_id": result.image_id,
                "frame_idx": frame_idx,
                "video_time_ms": timestamp_ms,
                "pred": result.plate_text,
                "ocr_text_raw": result.ocr_text_raw,
                "score": result.confidence,
                "det_score": result.det_score,
                "latency_ms": result.latency_ms,
                "bbox_xyxy": result.bbox_xyxy,
                "source": result.image_id,
            }
        )

        if writer is not None:
            vis = frame.copy()
            _draw_result(vis, result.plate_text, result.bbox_xyxy)
            writer.write(vis)

        processed += 1
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Processed frames: %d", processed)
    logging.info("JSON written: %s", args.output_json)
    if args.output_video is not None:
        logging.info("Annotated video written: %s", args.output_video)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
