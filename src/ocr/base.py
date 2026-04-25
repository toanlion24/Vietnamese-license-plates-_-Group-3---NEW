from __future__ import annotations

from typing import Protocol

import numpy as np

from src.utils.types import OcrResult, PlateCrop


class PlateOcr(Protocol):
    def recognize(self, plate_crop: PlateCrop, preprocessed: np.ndarray) -> OcrResult:
        ...


class DummyOcr:
    """Fallback OCR for pipeline integration tests."""

    def recognize(self, plate_crop: PlateCrop, preprocessed: np.ndarray) -> OcrResult:
        _ = preprocessed
        return OcrResult(
            image_id=plate_crop.image_id,
            text_raw="51H12345",
            text_norm="51H12345",
            ocr_score=0.5,
        )

