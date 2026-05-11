from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EasyOCR vs TrOCR comparison CSV into report-ready markdown.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("reports/video_compare_easyocr_vs_trocr.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/video_compare_summary.md"),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K frames by absolute latency gap.")
    return parser.parse_args()


def _to_float(value: str) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    pos = (len(values_sorted) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values_sorted[low]
    w = pos - low
    return values_sorted[low] * (1.0 - w) + values_sorted[high] * w


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x:.2f}"


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    input_csv = args.input_csv if args.input_csv.is_absolute() else (PROJECT_ROOT / args.input_csv)
    output_md = args.output_md if args.output_md.is_absolute() else (PROJECT_ROOT / args.output_md)

    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})

    if not rows:
        raise SystemExit(f"CSV rỗng: {input_csv}")

    same_count = sum(1 for r in rows if r.get("same_pred", "").strip().lower() == "true")
    total = len(rows)

    easy_lat = [_to_float(r.get("easyocr_latency_ms", "")) for r in rows]
    trocr_lat = [_to_float(r.get("trocr_latency_ms", "")) for r in rows]
    gap_lat = [_to_float(r.get("latency_gap_ms_trocr_minus_easyocr", "")) for r in rows]
    easy_vals = [x for x in easy_lat if x is not None]
    trocr_vals = [x for x in trocr_lat if x is not None]
    gap_vals = [x for x in gap_lat if x is not None]

    def stats(vals: list[float]) -> tuple[float | None, float | None, float | None]:
        if not vals:
            return (None, None, None)
        return (statistics.mean(vals), _pct(vals, 0.5), _pct(vals, 0.95))

    easy_mean, easy_p50, easy_p95 = stats(easy_vals)
    trocr_mean, trocr_p50, trocr_p95 = stats(trocr_vals)
    gap_mean, gap_p50, gap_p95 = stats(gap_vals)

    top_rows = []
    for r in rows:
        g = _to_float(r.get("latency_gap_ms_trocr_minus_easyocr", ""))
        if g is None:
            continue
        top_rows.append((abs(g), g, r))
    top_rows.sort(key=lambda x: x[0], reverse=True)
    top_rows = top_rows[: max(0, args.top_k)]

    lines: list[str] = [
        "# Tóm tắt so sánh EasyOCR vs TrOCR trên video",
        "",
        f"- Nguồn: `{input_csv}`",
        f"- Tổng số frame so sánh: **{total}**",
        f"- Tỉ lệ dự đoán giống nhau: **{same_count}/{total} = {(same_count / total * 100.0):.2f}%**",
        "",
        "## Bảng tổng hợp latency (ms)",
        "",
        "| Model/Chỉ số | Mean | P50 | P95 |",
        "| --- | ---: | ---: | ---: |",
        f"| EasyOCR latency | {_fmt(easy_mean)} | {_fmt(easy_p50)} | {_fmt(easy_p95)} |",
        f"| TrOCR latency | {_fmt(trocr_mean)} | {_fmt(trocr_p50)} | {_fmt(trocr_p95)} |",
        f"| Gap (TrOCR - EasyOCR) | {_fmt(gap_mean)} | {_fmt(gap_p50)} | {_fmt(gap_p95)} |",
        "",
        "## Top frame lệch latency lớn nhất",
        "",
        "| Rank | frame_idx | easyocr_pred | trocr_pred | easy_latency | trocr_latency | gap (trocr-easy) |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]

    for idx, (_, gap, row) in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{idx} | {row.get('frame_idx', '')} | {row.get('easyocr_pred', '')} | {row.get('trocr_pred', '')} | "
            f"{row.get('easyocr_latency_ms', '')} | {row.get('trocr_latency_ms', '')} | {gap:.4f} |"
        )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {output_md}")
    print(f"Same prediction rate: {(same_count / total * 100.0):.2f}%")


if __name__ == "__main__":
    main()
