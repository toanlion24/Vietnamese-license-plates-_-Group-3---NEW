"""Heuristic error labels for Buổi 4 prediction rows (GT vs pred + optional signals)."""

from __future__ import annotations

from src.eval.metrics_plate import levenshtein_chars
from src.postprocess.plate_rules import normalize_plate_text

# Labels requested for Buổi 4 reporting (mis-predictions; "ok" when normalized match).
ERROR_LABELS = frozenset(
    {
        "ok",
        "detect_miss",
        "bad_crop",
        "ocr_error",
        "postprocess_helped",
        "ambiguous_gt",
        # legacy bucket produced by older CSVs / fallbacks
        "ocr_or_spotting",
    }
)


def classify_plate_error(
    *,
    gt: str,
    pred: str,
    detect_hit: bool,
    ambiguous_gt: bool = False,
    ocr_raw_norm: str = "",
    pred_before_repair: str = "",
    det_score: float = 1.0,
    bad_crop_det_conf: float = 0.4,
) -> str:
    """Return an `error_type` string for one sample.

    * `ambiguous_gt` — set from manifest when the label is uncertain.
    * `postprocess_helped` — here: postprocess/repair *hurt* accuracy vs normalized
      OCR output (raw norm closer to GT than final pred). Kept name to match course wording.
    * `bad_crop` — heuristic when detection is weak (likely bad localization) and pred is wrong.
    """
    g = normalize_plate_text(gt)
    p = normalize_plate_text(pred)
    if not g and not p:
        return "ok"
    if g == p:
        return "ok"
    if ambiguous_gt:
        return "ambiguous_gt"
    if not detect_hit:
        return "detect_miss"

    raw_n = normalize_plate_text(ocr_raw_norm) if ocr_raw_norm else ""
    before_n = normalize_plate_text(pred_before_repair) if pred_before_repair else ""

    # If repair pipeline moved away from a better raw/normalized OCR hypothesis.
    if raw_n and p and raw_n != p:
        d_raw = levenshtein_chars(g, raw_n)
        d_final = levenshtein_chars(g, p)
        if d_raw < d_final:
            return "postprocess_helped"

    if before_n and p and before_n != p:
        d_before = levenshtein_chars(g, before_n)
        d_final = levenshtein_chars(g, p)
        if d_before < d_final:
            return "postprocess_helped"

    if det_score < bad_crop_det_conf:
        return "bad_crop"
    return "ocr_error"
