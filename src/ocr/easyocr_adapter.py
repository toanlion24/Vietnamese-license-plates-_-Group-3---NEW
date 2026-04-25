from __future__ import annotations

from src.postprocess.plate_rules import normalize_plate_text
from src.utils.types import OcrResult, PlateCrop

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None  # type: ignore[assignment]


class EasyOcrAdapter:
    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        if easyocr is None:
            raise ImportError("easyocr is required. Install with: pip install easyocr")
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def recognize(self, plate_crop: PlateCrop, preprocessed) -> OcrResult:  # noqa: ANN001
        results = self.reader.readtext(preprocessed, detail=1)
        if not results:
            text_raw = ""
            score = 0.0
        else:
            # (bbox, text, conf)
            _, text_raw, score = max(results, key=lambda item: float(item[2]))
            score = float(score)
        return OcrResult(
            image_id=plate_crop.image_id,
            text_raw=text_raw,
            text_norm=normalize_plate_text(text_raw),
            ocr_score=score,
        )

