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

