"""Phân tích lỗi theo từng ký tự và vùng trên biển (tỉnh / chữ / số serial).

Chỉ số ``index`` là trên chuỗi đã ``normalize_plate_text``. Vùng: ``index`` 0–1 → ``province``,
2 → ``letter``, còn lại → ``serial``. Chỉ số âm trả về ``unknown`` (không dùng trong lặp thông thường).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TypedDict

from src.postprocess.plate_rules import normalize_plate_text

REGION_PROVINCE = "province"
REGION_LETTER = "letter"
REGION_SERIAL = "serial"
REGION_UNKNOWN = "unknown"

CSV_FIELDS = [
    "image_id",
    "gt",
    "pred",
    "index",
    "region",
    "gt_char",
    "pred_char",
    "match",
]


class CharErrorRow(TypedDict):
    image_id: str
    gt: str
    pred: str
    index: int
    region: str
    gt_char: str
    pred_char: str
    match: bool


def region_for_index(index: int) -> str:
    """Gán vùng theo chỉ số ký tự sau chuẩn hóa (0-based)."""
    if index < 0:
        return REGION_UNKNOWN
    if index < 2:
        return REGION_PROVINCE
    if index == 2:
        return REGION_LETTER
    return REGION_SERIAL


def character_errors_by_region(image_id: str, gt: str, pred: str) -> list[CharErrorRow]:
    """
    So khớp từng vị trí (căn trái). Khi một chuỗi ngắn hơn, ký tự thiếu coi như rỗng.

    Một hàng cho mỗi chỉ số từ 0 đến ``max(len(gt), len(pred)) - 1`` (trên bản đã normalize).
    """
    g = normalize_plate_text(gt)
    p = normalize_plate_text(pred)
    n = max(len(g), len(p))
    out: list[CharErrorRow] = []
    for i in range(n):
        gc = g[i] if i < len(g) else ""
        pc = p[i] if i < len(p) else ""
        reg = region_for_index(i)
        out.append(
            {
                "image_id": image_id,
                "gt": g,
                "pred": p,
                "index": i,
                "region": reg,
                "gt_char": gc,
                "pred_char": pc,
                "match": gc == pc,
            }
        )
    return out


def prediction_rows_to_char_error_rows(
    rows: list[dict[str, str]],
    *,
    id_keys: tuple[str, ...] = ("image_id", "id", "filename"),
    gt_keys: tuple[str, ...] = ("gt", "text_gt", "label"),
    pred_keys: tuple[str, ...] = ("pred", "text_pred", "plate_text", "prediction"),
) -> list[dict[str, Any]]:
    """Nối các dòng lỗi ký tự cho toàn bộ batch prediction (mỗi mẫu nhiều dòng theo index)."""
    result: list[dict[str, Any]] = []

    def pick(row: dict[str, str], keys: tuple[str, ...]) -> str:
        for k in keys:
            v = (row.get(k) or "").strip()
            if v:
                return v
        return ""

    for row in rows:
        iid = pick(row, id_keys)
        if not iid:
            continue
        g = pick(row, gt_keys)
        pr = pick(row, pred_keys)
        for block in character_errors_by_region(iid, g, pr):
            result.append(
                {
                    "image_id": block["image_id"],
                    "gt": block["gt"],
                    "pred": block["pred"],
                    "index": block["index"],
                    "region": block["region"],
                    "gt_char": block["gt_char"],
                    "pred_char": block["pred_char"],
                    "match": block["match"],
                }
            )
    return result


def export_character_errors_csv(
    prediction_csv: Path,
    output_csv: Path,
) -> int:
    """
    Đọc CSV prediction (có ``image_id``, ``gt``, ``pred``), ghi CSV chi tiết lỗi theo ký tự.

    Trả về số **dòng** đã ghi (không tính header).
    """
    raw_rows: list[dict[str, str]] = []
    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            raw_rows.append({k: (v or "") for k, v in row.items()})

    flat = prediction_rows_to_char_error_rows(raw_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        w.writeheader()
        for item in flat:
            w.writerow(
                {
                    "image_id": item["image_id"],
                    "gt": item["gt"],
                    "pred": item["pred"],
                    "index": item["index"],
                    "region": item["region"],
                    "gt_char": item["gt_char"],
                    "pred_char": item["pred_char"],
                    "match": str(item["match"]).lower(),
                }
            )
    return len(flat)
