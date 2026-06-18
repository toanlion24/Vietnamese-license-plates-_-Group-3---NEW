from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from src.utils.types import FrameData

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


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


class ImageReader:
    """Reader for single images."""

    def read_image(self, path: Path) -> FrameData:
        """Read a single image and return FrameData."""
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read image: {path}")
        return FrameData(
            image_id=path.stem,
            frame=frame,
            source=str(path),
        )


class VideoReader:
    """Reader for video files - yields frames one by one."""

    def __init__(self, video_path: Path, *, skip_frames: int = 0) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_idx = 0
        self.skip_frames = skip_frames

    def read_frame(self) -> FrameData | None:
        """Read next frame from video. Returns None when video ends."""
        # Skip frames if requested
        while self.skip_frames > 0 and self.frame_idx % (self.skip_frames + 1) != 0:
            self.cap.grab()
            self.frame_idx += 1

        ret, frame = self.cap.read()
        if not ret:
            return None

        timestamp_ms = (self.frame_idx / self.fps) * 1000 if self.fps > 0 else 0
        frame_data = FrameData(
            image_id=f"video_{self.frame_idx:06d}",
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            source=str(self.video_path),
            timestamp_ms=timestamp_ms,
        )
        self.frame_idx += 1
        return frame_data

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate over all frames."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        """Release video capture."""
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args) -> None:
        self.release()

