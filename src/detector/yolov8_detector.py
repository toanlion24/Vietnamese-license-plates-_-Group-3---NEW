from __future__ import annotations

from pathlib import Path

from src.utils.types import Detection, FrameData

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]


class YoloV8PlateDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        class_name: str = "license_plate",
    ) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.class_name = class_name

    def predict(self, frame_data: FrameData) -> list[Detection]:
        results = self.model.predict(frame_data.frame, conf=self.conf_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                score = float(box.conf[0].item())
                detections.append(
                    Detection(
                        image_id=frame_data.image_id,
                        bbox_xyxy=(x1, y1, x2, y2),
                        score=score,
                        class_name=self.class_name,
                    )
                )
        return detections

