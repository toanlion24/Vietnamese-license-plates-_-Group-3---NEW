from __future__ import annotations

import re

PLATE_PATTERN = re.compile(r"^[0-9]{2}[A-Z][0-9]{4,5}$")


def normalize_plate_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned


def is_valid_vn_plate(text: str) -> bool:
    return bool(PLATE_PATTERN.match(text))


def repair_common_ocr_errors(text: str) -> str:
    normalized = normalize_plate_text(text)
    if len(normalized) >= 2:
        normalized = normalized[:2].replace("O", "0") + normalized[2:]
    return normalized


def repair_tail_digit_confusions(text: str) -> str:
    """Sửa ký tự dễ nhầm ở phần sau chữ cái tỉnh (index >= 3). Đầu vào nên đã normalize."""
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
    base = repair_common_ocr_errors(text)
    if aggressive_tail:
        return repair_tail_digit_confusions(base)
    return base

