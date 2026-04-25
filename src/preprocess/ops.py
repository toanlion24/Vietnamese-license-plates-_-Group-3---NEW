from __future__ import annotations

import cv2
import numpy as np

from src.utils.types import Detection, FrameData, PlateCrop


def crop_plate(frame_data: FrameData, detection: Detection) -> PlateCrop:
    x1, y1, x2, y2 = detection.bbox_xyxy
    crop = frame_data.frame[y1:y2, x1:x2].copy()
    return PlateCrop(
        image_id=frame_data.image_id,
        crop=crop,
        bbox_xyxy=detection.bbox_xyxy,
        det_score=detection.score,
    )


def preprocess_plate(crop: np.ndarray, output_size: tuple[int, int] = (320, 120)) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_CUBIC)
    normalized = cv2.equalizeHist(resized)
    return normalized

