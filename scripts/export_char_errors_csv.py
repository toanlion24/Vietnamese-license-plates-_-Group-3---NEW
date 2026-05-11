"""Xuất CSV phân tích lỗi theo ký tự và vùng từ file prediction Buổi 4 / eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.char_error_regions import export_character_errors_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred-csv",
        type=Path,
        required=True,
        help="CSV có image_id, gt, pred (như outputs/buoi4/*_predictions.csv).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/char_errors_by_region.csv"),
    )
    return p.parse_args()


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args()
    n = export_character_errors_csv(args.pred_csv, args.output_csv)
    print(f"Đã ghi {n} dòng tại {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
