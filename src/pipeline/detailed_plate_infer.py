"""Single-image inference with extra OCR/postprocess fields for error analysis."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from src.detector.base import PlateDetector
from src.ocr.base import PlateOcr
from src.postprocess.plate_rules import normalize_plate_text, postprocess_plate_text
from src.preprocess.ops import crop_plate, preprocess_plate
from src.utils.types import FrameData


@dataclass(slots=True)
class DetailedPlateInferResult:
    image_id: str
    plate_text: str
    bbox_xyxy: tuple[int, int, int, int] | None
    confidence: float
    latency_ms: float
    det_score: float
    ocr_text_raw: str
    text_before_repair: str


def infer_plate_detailed(
    detector: PlateDetector,
    ocr: PlateOcr,
    frame_data: FrameData,
    *,
    crop_margin_ratio: float = 0.0,
    preprocess_clahe: bool = False,
    aggressive_postprocess: bool = False,
) -> DetailedPlateInferResult:
    start = perf_counter()
    detections = detector.predict(frame_data)
    if not detections:
        elapsed = (perf_counter() - start) * 1000.0
        return DetailedPlateInferResult(
            image_id=frame_data.image_id,
            plate_text="",
            bbox_xyxy=None,
            confidence=0.0,
            latency_ms=elapsed,
            det_score=0.0,
            ocr_text_raw="",
            text_before_repair="",
        )

    best = max(detections, key=lambda d: d.score)
    plate_crop = crop_plate(frame_data, best, margin_ratio=crop_margin_ratio)
    prepared = preprocess_plate(plate_crop.crop, use_clahe=preprocess_clahe)
    ocr_out = ocr.recognize(plate_crop, prepared)
    before_repair = ocr_out.text_norm or normalize_plate_text(ocr_out.text_raw)
    final_text = postprocess_plate_text(before_repair, aggressive_tail=aggressive_postprocess)
    elapsed = (perf_counter() - start) * 1000.0
    conf = min(best.score, ocr_out.ocr_score)

    return DetailedPlateInferResult(
        image_id=frame_data.image_id,
        plate_text=final_text,
        bbox_xyxy=best.bbox_xyxy,
        confidence=float(conf),
        latency_ms=elapsed,
        det_score=float(best.score),
        ocr_text_raw=ocr_out.text_raw,
        text_before_repair=before_repair,
    )
