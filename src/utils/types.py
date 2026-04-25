from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class FrameData:
    image_id: str
    frame: np.ndarray
    source: str
    timestamp_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Detection:
    image_id: str
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    class_name: str = "license_plate"


@dataclass(slots=True)
class PlateCrop:
    image_id: str
    crop: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    det_score: float


@dataclass(slots=True)
class OcrResult:
    image_id: str
    text_raw: str
    text_norm: str
    ocr_score: float


@dataclass(slots=True)
class PipelineResult:
    image_id: str
    plate_text: str
    bbox_xyxy: tuple[int, int, int, int] | None
    confidence: float
    source: str
    timestamp_ms: float | None = None


@dataclass(slots=True)
class EvalRecord:
    image_id: str
    gt: str
    pred: str
    error_type: str


@dataclass(slots=True)
class PathConfig:
    root: Path
    input_dir: Path
    output_dir: Path

