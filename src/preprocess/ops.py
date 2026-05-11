from __future__ import annotations

import cv2
import numpy as np

from src.utils.types import Detection, FrameData, PlateCrop


def crop_plate(
    frame_data: FrameData,
    detection: Detection,
    *,
    margin_ratio: float = 0.0,
) -> PlateCrop:
    x1, y1, x2, y2 = detection.bbox_xyxy
    if margin_ratio > 0:
        h, w = frame_data.frame.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(bw * margin_ratio)
        my = int(bh * margin_ratio)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)
    crop = frame_data.frame[y1:y2, x1:x2].copy()
    return PlateCrop(
        image_id=frame_data.image_id,
        crop=crop,
        bbox_xyxy=(x1, y1, x2, y2),
        det_score=detection.score,
    )


def preprocess_plate(
    crop: np.ndarray,
    output_size: tuple[int, int] = (320, 120),
    *,
    use_clahe: bool = False,
) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_CUBIC)
    if use_clahe:
        return resized
    return cv2.equalizeHist(resized)

