from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frame-level OCR outputs from two video JSON files.")
    parser.add_argument("--easyocr-json", type=Path, required=True)
    parser.add_argument("--trocr-json", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/video_compare_easyocr_vs_trocr.csv"),
    )
    return parser.parse_args()


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_by_frame(path: Path) -> dict[int, dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        frame_idx = row.get("frame_idx")
        try:
            fi = int(frame_idx)
        except (TypeError, ValueError):
            continue
        out[fi] = row
    return out


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    easy_map = _load_by_frame(args.easyocr_json)
    trocr_map = _load_by_frame(args.trocr_json)
    all_frames = sorted(set(easy_map) | set(trocr_map))

    output_csv = args.output_csv if args.output_csv.is_absolute() else (PROJECT_ROOT / args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "frame_idx",
        "video_time_ms",
        "easyocr_pred",
        "trocr_pred",
        "same_pred",
        "easyocr_latency_ms",
        "trocr_latency_ms",
        "latency_gap_ms_trocr_minus_easyocr",
        "easyocr_score",
        "trocr_score",
        "easyocr_bbox",
        "trocr_bbox",
    ]

    matched = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for fi in all_frames:
            e = easy_map.get(fi, {})
            t = trocr_map.get(fi, {})
            e_pred = _norm_text(e.get("pred", ""))
            t_pred = _norm_text(t.get("pred", ""))
            same = e_pred == t_pred
            if same:
                matched += 1

            e_lat = _to_float(e.get("latency_ms"))
            t_lat = _to_float(t.get("latency_ms"))
            gap = (t_lat - e_lat) if (t_lat is not None and e_lat is not None) else None

            time_ms = e.get("video_time_ms", t.get("video_time_ms", ""))
            writer.writerow(
                {
                    "frame_idx": fi,
                    "video_time_ms": time_ms,
                    "easyocr_pred": e_pred,
                    "trocr_pred": t_pred,
                    "same_pred": str(same).lower(),
                    "easyocr_latency_ms": "" if e_lat is None else f"{e_lat:.4f}",
                    "trocr_latency_ms": "" if t_lat is None else f"{t_lat:.4f}",
                    "latency_gap_ms_trocr_minus_easyocr": "" if gap is None else f"{gap:.4f}",
                    "easyocr_score": e.get("score", ""),
                    "trocr_score": t.get("score", ""),
                    "easyocr_bbox": e.get("bbox_xyxy", ""),
                    "trocr_bbox": t.get("bbox_xyxy", ""),
                }
            )

    print(f"Wrote: {output_csv}")
    print(f"Frames compared: {len(all_frames)}")
    print(f"Same prediction count: {matched}")


if __name__ == "__main__":
    main()
