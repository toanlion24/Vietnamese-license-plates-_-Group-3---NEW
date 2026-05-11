"""EasyOCR + TrOCR ensemble: chọn nhánh theo định dạng biển VN (infer không cần GT)."""

from __future__ import annotations

from src.ocr.easyocr_adapter import EasyOcrAdapter
from src.ocr.trocr_adapter import TrOcrAdapter
from src.postprocess.plate_rules import (
    is_valid_vn_plate,
    normalize_plate_text,
    postprocess_plate_text,
)
from src.utils.types import OcrResult, PlateCrop


class EasyTrocrEnsembleOcr:
    """Chạy hai OCR trên cùng ``preprocessed``; trả về ``OcrResult`` của một nhánh (pipeline hậu xử lý thêm một lần)."""

    def __init__(
        self,
        easy: EasyOcrAdapter,
        trocr: TrOcrAdapter,
        *,
        aggressive_post: bool = False,
    ) -> None:
        self.easy = easy
        self.trocr = trocr
        self.aggressive_post = aggressive_post

    def _final(self, r: OcrResult) -> str:
        return postprocess_plate_text(
            r.text_norm or normalize_plate_text(r.text_raw),
            aggressive_tail=self.aggressive_post,
        )

    def recognize(self, plate_crop: PlateCrop, preprocessed) -> OcrResult:  # noqa: ANN001
        r_e = self.easy.recognize(plate_crop, preprocessed)
        r_t = self.trocr.recognize(plate_crop, preprocessed)

        t_e, t_t = self._final(r_e), self._final(r_t)

        if t_e == t_t:
            score = max(float(r_e.ocr_score), float(r_t.ocr_score))
            return OcrResult(
                image_id=plate_crop.image_id,
                text_raw=r_t.text_raw,
                text_norm=r_t.text_norm,
                ocr_score=min(1.0, score),
            )

        ok_e, ok_t = is_valid_vn_plate(t_e), is_valid_vn_plate(t_t)
        if ok_e and not ok_t:
            return r_e
        if ok_t and not ok_e:
            return r_t
        if ok_e and ok_t:
            return r_t if len(t_t) >= len(t_e) else r_e
        return r_t if float(r_t.ocr_score) >= float(r_e.ocr_score) else r_e
