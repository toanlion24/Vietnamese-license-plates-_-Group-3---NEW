from __future__ import annotations

from time import perf_counter

from src.detector.base import PlateDetector
from src.ocr.base import PlateOcr
from src.postprocess.plate_rules import normalize_plate_text, postprocess_plate_text
from src.preprocess.ops import crop_plate, preprocess_plate
from src.utils.types import FrameData, PipelineResult


class PlateInferencePipeline:
    def __init__(
        self,
        detector: PlateDetector,
        ocr: PlateOcr,
        *,
        crop_margin_ratio: float = 0.0,
        preprocess_clahe: bool = False,
        aggressive_postprocess: bool = False,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.crop_margin_ratio = crop_margin_ratio
        self.preprocess_clahe = preprocess_clahe
        self.aggressive_postprocess = aggressive_postprocess

    def run(self, frame_data: FrameData) -> PipelineResult:
        start = perf_counter()
        detections = self.detector.predict(frame_data)
        if not detections:
            return PipelineResult(
                image_id=frame_data.image_id,
                plate_text="",
                bbox_xyxy=None,
                confidence=0.0,
                source=frame_data.source,
                timestamp_ms=frame_data.timestamp_ms,
            )

        best_detection = max(detections, key=lambda d: d.score)
        plate_crop = crop_plate(frame_data, best_detection, margin_ratio=self.crop_margin_ratio)
        prepared = preprocess_plate(plate_crop.crop, use_clahe=self.preprocess_clahe)
        ocr_out = self.ocr.recognize(plate_crop, prepared)
        text = postprocess_plate_text(
            ocr_out.text_norm or normalize_plate_text(ocr_out.text_raw),
            aggressive_tail=self.aggressive_postprocess,
        )
        elapsed = (perf_counter() - start) * 1000.0

        return PipelineResult(
            image_id=frame_data.image_id,
            plate_text=text,
            bbox_xyxy=best_detection.bbox_xyxy,
            confidence=min(best_detection.score, ocr_out.ocr_score),
            source=frame_data.source,
            timestamp_ms=elapsed,
        )

