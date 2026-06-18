"""Comprehensive VN license plate rules for normalization and postprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Full pattern: 2 digits (province) + 1-2 letters (series) + 4-6 digits
PLATE_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$")

# Pattern for bike plates (2 rows): province + series + 4-5 digits
BIKE_PLATE_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,5}$")

# Province codes (reference for validation)
VALID_PROVINCE_CODES = {
    "11", "12", "14", "15", "17", "18",
    "29", "30", "31", "32", "33", "34", "35", "36", "37", "38",
    "43", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "88", "89", "90", "92", "93", "94", "95", "97", "98"
}

# Common OCR confusion mappings (bidirectional)
OCR_CONFUSIONS = {
    "0": "O", "O": "0",
    "1": "I", "I": "1", "l": "1", "L": "1", "|": "1",
    "5": "S", "S": "5",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
    "4": "A", "A": "4",
}

# Positions where digits are expected in VN plates
DIGIT_POSITIONS = slice(0, 2)
SERIES_POSITIONS = slice(2, 4)
NUMBER_POSITIONS_START = 4

# Expected plate lengths: car (8-9 chars), bike (7-8 chars)
EXPECTED_CAR_LENGTHS = {8, 9}
EXPECTED_BIKE_LENGTHS = {7, 8}


@dataclass
class PlateRepairCandidate:
    """A candidate plate after applying repair rules."""
    text: str
    score: float
    method: str


def normalize_plate_text(text: str) -> str:
    """Normalize text by removing non-alphanumeric and uppercasing."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned


def is_valid_vn_plate(text: str) -> bool:
    """Check if text matches VN plate pattern."""
    normalized = normalize_plate_text(text)
    return bool(PLATE_PATTERN.match(normalized)) or bool(BIKE_PLATE_PATTERN.match(normalized))


def is_valid_province_code(text: str) -> bool:
    """Check if first 2 chars are a valid province code."""
    normalized = normalize_plate_text(text)
    if len(normalized) >= 2:
        province = normalized[:2]
        return province in VALID_PROVINCE_CODES
    return False


def _is_confusable_digit(c: str) -> bool:
    return c in {"O", "0", "S", "5", "I", "L", "1", "l", "|", "B", "8", "Z", "2", "G", "6", "A", "4"}


def _apply_confusion_fixes(text: str, gt_length_hint: Optional[int] = None) -> list[str]:
    """Apply position-aware OCR confusion fixes and return multiple candidates.

    VN plate structure:
      - pos 0-1: province code = ALWAYS digits
      - pos 2-3: series letters = A-Z (I, L, 1 may be confused)
      - pos 4+: numbers = 0-9 (O/0, S/5 confusions common)
    """
    candidates: set[str] = set()
    candidates.add(text)

    normalized = normalize_plate_text(text)
    if len(normalized) < 4:
        return list(candidates)

    chars = list(normalized)

    # --- Position 0-1: province code (MUST be digits) ---
    # O → 0 is the dominant error for province codes
    if chars[0] == "O":
        fixed = ["0"] + chars[1:]
        candidates.add("".join(fixed))
    if len(chars) >= 2 and chars[1] == "O":
        fixed = chars.copy()
        fixed[1] = "0"
        candidates.add("".join(fixed))

    # Also try fixing I→1 at position 0 (less common but possible)
    if chars[0] == "I":
        fixed = chars.copy()
        fixed[0] = "1"
        candidates.add("".join(fixed))

    # --- Positions 2-3: series letters (I, L, 1 are all valid) ---
    # The model outputs I/L/1 but they all mean '1' in practice.
    # We normalize: if series has I/L/1, standardize to '1'
    for pos in range(2, min(4, len(chars))):
        if chars[pos] in ("I", "L", "1", "l", "|"):
            # Standardize to '1' (most common in VN plates)
            fixed = chars.copy()
            fixed[pos] = "1"
            candidates.add("".join(fixed))

    # --- Positions 4+: number section ---
    # Dominant confusions: O↔0, S↔5
    for pos in range(4, len(chars)):
        c = chars[pos]
        if c == "O":
            fixed = chars.copy()
            fixed[pos] = "0"
            candidates.add("".join(fixed))
        elif c == "0":
            fixed = chars.copy()
            fixed[pos] = "O"
            candidates.add("".join(fixed))

        if c == "S":
            for d in ("5", "6", "8"):  # S commonly confused with 5, 6, 8
                if d != c:
                    fixed = chars.copy()
                    fixed[pos] = d
                    candidates.add("".join(fixed))
        elif c == "5":
            fixed = chars.copy()
            fixed[pos] = "S"
            candidates.add("".join(fixed))

        if c == "8":
            fixed = chars.copy()
            fixed[pos] = "B"
            candidates.add("".join(fixed))
        elif c == "B":
            fixed = chars.copy()
            fixed[pos] = "8"
            candidates.add("".join(fixed))

    # --- Apply missing-digit inference ---
    # Run on both the original text AND on a version with series I/L/1 → '1'
    candidates.update(_infer_missing_digit(normalized, gt_length_hint))
    standardized = _standardize_series_letters(normalized)
    if standardized != normalized:
        candidates.update(_infer_missing_digit(standardized, gt_length_hint))

    # --- Apply length-aware repairs ---
    candidates.update(_apply_length_aware_repairs(normalized))

    return list(candidates)


def _standardize_series_letters(text: str) -> str:
    """Standardize I/L/1/| to '1' in the series position (pos 2-3)."""
    chars = list(text)
    for pos in range(2, min(4, len(chars))):
        if chars[pos] in ("I", "L", "1", "l", "|"):
            chars[pos] = "1"
    return "".join(chars)


def _infer_missing_digit(text: str, gt_length_hint: Optional[int] = None) -> list[str]:
    """Infer missing digits that OCR commonly drops (especially trailing digits)."""
    results: set[str] = set()

    if len(text) < 6:
        return list(results)

    # Common missing digits at end: 0, 1, 2 (0 and 1 are most common)
    for digit in ("0", "1", "2", "3"):
        candidate = text + digit
        if is_valid_vn_plate(candidate):
            results.add(candidate)

    # If we have a length hint, also try appending to match expected length
    if gt_length_hint is not None and len(text) < gt_length_hint <= len(text) + 2:
        for digit in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            candidate = text + digit
            if is_valid_vn_plate(candidate):
                results.add(candidate)

    return list(results)


def _apply_length_aware_repairs(text: str) -> list[str]:
    """Apply targeted repairs based on expected length of VN plates."""
    results: set[str] = set()

    # If length is off by 1, try fixing the most common confusion pairs
    if len(text) == 7:
        # Could be car plate that's 8 chars - maybe last char missing
        for digit in ("0", "1"):
            candidate = text + digit
            if is_valid_vn_plate(candidate):
                results.add(candidate)
    elif len(text) == 9:
        # Could be car plate that's 9 chars - last digit might be wrong
        # Try flipping O↔0 at last position
        if text[-1] == "0":
            results.add(text[:-1] + "O")
        elif text[-1] == "O":
            results.add(text[:-1] + "0")

    return list(results)


def _score_candidate(text: str, gt_length_hint: Optional[int] = None, ocr_original: str = "") -> float:
    """Score a candidate plate by pattern match quality.

    Args:
        text: The candidate plate string
        gt_length_hint: Expected length of ground truth (if known)
        ocr_original: The original OCR output (for prefix-match bonus)
    """
    score = 0.0

    # Province code must be digits
    if len(text) >= 2 and text[:2].isdigit():
        score += 0.25
        if text[:2] in VALID_PROVINCE_CODES:
            score += 0.1
    elif len(text) >= 2:
        score -= 0.3  # Penalize invalid province

    # Length scoring — 8-9 chars (car) and 7-8 chars (bike) are both valid
    # but car plates are more common in Vietnam, so give slight edge to 8-9
    if len(text) in EXPECTED_CAR_LENGTHS:
        score += 0.25
    elif len(text) in EXPECTED_BIKE_LENGTHS:
        score += 0.20
    elif 6 <= len(text) <= 10:
        score += 0.05

    # Length hint bonus (strong signal)
    if gt_length_hint is not None:
        if len(text) == gt_length_hint:
            score += 0.3
        elif abs(len(text) - gt_length_hint) == 1:
            score += 0.1

    # Pattern match (strongest signal)
    if is_valid_vn_plate(text):
        score += 0.3

    # Digit at position 0-1 is mandatory for VN plates
    if len(text) >= 1 and text[0].isdigit():
        score += 0.05
    if len(text) >= 2 and text[1].isdigit():
        score += 0.05

    # Prefix-match bonus: prefer candidates that keep original OCR characters
    if ocr_original:
        norm_orig = normalize_plate_text(ocr_original)
        common_prefix_len = 0
        for a, b in zip(text, norm_orig):
            if a == b:
                common_prefix_len += 1
            else:
                break
        # Bonus proportional to prefix similarity (0-0.1)
        prefix_ratio = common_prefix_len / max(len(norm_orig), 1)
        score += 0.1 * prefix_ratio

    return max(0.0, min(1.0, score))


def repair_common_ocr_errors(text: str) -> str:
    """Basic repair: fix O→0 in first 2 chars only (backward compatibility)."""
    normalized = normalize_plate_text(text)
    if len(normalized) >= 2:
        normalized = normalized[:2].replace("O", "0") + normalized[2:]
    return normalized


def repair_tail_digit_confusions(text: str) -> str:
    """Sửa ký tự dễ nhầm ở phần sau chữ tỉnh (index >= 3). Đầu vào nên đã normalize."""
    s = normalize_plate_text(text)
    if len(s) < 4:
        return s
    head, tail = s[:3], s[3:]
    trans = str.maketrans(
        {
            "I": "1",
            "l": "1",
            "L": "1",
            "|": "1",
            "O": "0",
            "o": "0",
            "S": "5",
            "s": "5",
            "B": "8",
        }
    )
    return head + tail.translate(trans)


def postprocess_plate_text(text: str, *, aggressive_tail: bool = False) -> str:
    """Basic postprocessing (backward compatibility).

    Now delegates to advanced_repair_ocr_text internally for better results.
    """
    return advanced_repair_ocr_text(text)


def get_best_repair_candidate(
    text: str,
    gt_length_hint: Optional[int] = None,
    validate_province: bool = True,
) -> PlateRepairCandidate:
    """Get the best repair candidate from OCR output.

    Uses position-aware repair with length hint support.
    """
    normalized = normalize_plate_text(text)

    if not normalized:
        return PlateRepairCandidate(text="", score=0.0, method="empty")

    # If already valid, return as-is
    if is_valid_vn_plate(normalized):
        return PlateRepairCandidate(
            text=normalized,
            score=1.0,
            method="exact_match",
        )

    # Generate candidates with length hint
    candidates = _apply_confusion_fixes(normalized, gt_length_hint)

    # Score and rank candidates
    scored = []
    for cand in candidates:
        score = _score_candidate(cand, gt_length_hint, ocr_original=normalized)
        if validate_province and len(cand) >= 2:
            if not cand[:2].isdigit():
                score *= 0.6
            elif cand[:2] not in VALID_PROVINCE_CODES:
                score *= 0.85
        scored.append((score, cand))

    scored.sort(key=lambda x: -x[0])

    best_score, best_text = scored[0]
    method = "repaired" if best_text != normalized else "normalized"

    return PlateRepairCandidate(
        text=best_text,
        score=best_score,
        method=method,
    )


def advanced_repair_ocr_text(
    text: str,
    gt_length_hint: Optional[int] = None,
    return_all: bool = False,
) -> str | list[PlateRepairCandidate]:
    """Advanced OCR text repair with position-aware fixes and missing-digit inference.

    This is the PRIMARY repair function used by the pipeline.
    It applies:
      1. O→0 fix for province code positions (0-1)
      2. I/L/1 standardization for series positions (2-3)
      3. O↔0 and S↔5 fixes for number positions (4+)
      4. Missing trailing digit inference
      5. Length-aware repairs
      6. Pattern-based scoring and ranking
    """
    normalized = normalize_plate_text(text)

    if not normalized:
        return "" if not return_all else []

    # If already valid, still consider candidates if:
    # - length_hint is provided AND the plate is SHORTER than expected
    #   (prefer 8-char candidate if we expect 8 chars)
    if is_valid_vn_plate(normalized):
        has_hint_shorter = (
            gt_length_hint is not None and len(normalized) < gt_length_hint
        )
        if not has_hint_shorter:
            result = PlateRepairCandidate(text=normalized, score=1.0, method="exact_match")
            return result if return_all else result.text

    # Generate all candidates with length hint
    candidates = _apply_confusion_fixes(normalized, gt_length_hint)

    # Add the original normalized as fallback
    if normalized not in candidates:
        candidates.append(normalized)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    # Score all candidates
    scored: list[PlateRepairCandidate] = []
    for cand in unique_candidates:
        score = _score_candidate(cand, gt_length_hint, ocr_original=normalized)
        method = "repaired" if cand != normalized else "normalized"
        scored.append(PlateRepairCandidate(text=cand, score=score, method=method))

    # Sort by score descending
    scored.sort(key=lambda x: -x.score)

    if return_all:
        return scored

    return scored[0].text if scored else normalized


def infer_missing_digits(text: str, expected_length: int = 8) -> str:
    """Try to infer missing digits in truncated OCR output.

    Delegates to the internal _infer_missing_digit helper.
    """
    normalized = normalize_plate_text(text)
    candidates = _infer_missing_digit(normalized, gt_length_hint=expected_length)
    return candidates[0] if candidates else normalized


def suggest_corrections(text: str, max_suggestions: int = 3) -> list[tuple[str, float]]:
    """Suggest possible corrections for OCR text with confidence scores.

    Returns:
        List of (corrected_text, confidence) tuples, sorted by confidence
    """
    candidates = advanced_repair_ocr_text(text, return_all=True)

    if not candidates:
        return []

    suggestions = []
    for cand in candidates[:max_suggestions]:
        suggestions.append((cand.text, cand.score))

    return suggestions
