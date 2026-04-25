from __future__ import annotations

from time import perf_counter

from src.detector.base import PlateDetector
from src.ocr.base import PlateOcr
from src.postprocess.plate_rules import normalize_plate_text, repair_common_ocr_errors
from src.preprocess.ops import crop_plate, preprocess_plate
from src.utils.types import FrameData, PipelineResult


class PlateInferencePipeline:
    def __init__(self, detector: PlateDetector, ocr: PlateOcr) -> None:
        self.detector = detector
        self.ocr = ocr

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
        plate_crop = crop_plate(frame_data, best_detection)
        prepared = preprocess_plate(plate_crop.crop)
        ocr_out = self.ocr.recognize(plate_crop, prepared)
        text = repair_common_ocr_errors(ocr_out.text_norm or normalize_plate_text(ocr_out.text_raw))
        elapsed = (perf_counter() - start) * 1000.0

        return PipelineResult(
            image_id=frame_data.image_id,
            plate_text=text,
            bbox_xyxy=best_detection.bbox_xyxy,
            confidence=min(best_detection.score, ocr_out.ocr_score),
            source=frame_data.source,
            timestamp_ms=elapsed,
        )

