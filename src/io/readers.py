from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from src.utils.types import FrameData

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(input_dir: Path) -> Iterator[FrameData]:
    for path in sorted(input_dir.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        yield FrameData(image_id=path.stem, frame=frame, source=str(path))

