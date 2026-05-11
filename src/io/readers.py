from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from src.utils.types import FrameData

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(input_dir: Path) -> Iterator[FrameData]:
    root = input_dir.resolve()
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        rel = path.relative_to(root)
        image_id = str(rel.with_suffix("")).replace("\\", "/")
        yield FrameData(image_id=image_id, frame=frame, source=str(path))

