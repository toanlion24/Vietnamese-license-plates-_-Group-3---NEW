from __future__ import annotations

from typing import Protocol

from src.utils.types import Detection, FrameData


class PlateDetector(Protocol):
    def predict(self, frame_data: FrameData) -> list[Detection]:
        ...


class DummyCenterDetector:
    """Fallback detector for wiring and local testing."""

    def predict(self, frame_data: FrameData) -> list[Detection]:
        h, w = frame_data.frame.shape[:2]
        x1, y1 = int(w * 0.25), int(h * 0.40)
        x2, y2 = int(w * 0.75), int(h * 0.65)
        return [
            Detection(
                image_id=frame_data.image_id,
                bbox_xyxy=(x1, y1, x2, y2),
                score=0.5,
            )
        ]

