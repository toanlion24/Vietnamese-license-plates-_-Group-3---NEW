from __future__ import annotations

from typing import Protocol, Union

import numpy as np
from PIL.Image import Image as PILImage

from src.utils.types import OcrResult, PlateCrop


class PlateOcr(Protocol):
    def recognize(self, plate_crop: PlateCrop, preprocessed: Union[np.ndarray, PILImage]) -> OcrResult:
        ...


class DummyOcr:
    """Fallback OCR for pipeline integration tests."""

    def recognize(self, plate_crop: PlateCrop, preprocessed=None) -> OcrResult:
        return OcrResult(
            image_id=plate_crop.image_id,
            text_raw="DETECTING",
            text_norm="DETECTING",
            ocr_score=0.5,
        )


class FastDummyOcr:
    """Instant OCR - returns 'Detecting...' for fast detection-only mode."""

    def recognize(self, plate_crop: PlateCrop, preprocessed=None) -> OcrResult:
        return OcrResult(
            image_id=plate_crop.image_id,
            text_raw="Detecting...",
            text_norm="Detecting...",
            ocr_score=1.0,
        )

