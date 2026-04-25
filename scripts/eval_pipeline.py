from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json

from src.eval.metrics_plate import cer, export_records, plate_accuracy, wer
from src.postprocess.plate_rules import normalize_plate_text
from src.utils.types import EvalRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR outputs using CER/WER/plate accuracy.")
    parser.add_argument("--pred-csv", type=Path, required=True, help="CSV with columns: image_id,gt,pred")
    parser.add_argument("--report-json", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--errors-csv", type=Path, default=Path("reports/error_records.csv"))
    return parser.parse_args()


def load_records(pred_csv: Path) -> list[EvalRecord]:
    rows: list[EvalRecord] = []
    with pred_csv.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            gt = row.get("gt", "")
            pred = row.get("pred", "")
            rows.append(
                EvalRecord(
                    image_id=row.get("image_id", ""),
                    gt=gt,
                    pred=pred,
                    error_type="ok" if normalize_plate_text(gt) == normalize_plate_text(pred) else "ocr_or_postprocess",
                )
            )
    return rows


if __name__ == "__main__":
    args = parse_args()
    records = load_records(args.pred_csv)
    metrics = {
        "cer": cer(records),
        "wer": wer(records),
        "plate_accuracy": plate_accuracy(records),
        "num_samples": len(records),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_records(records, args.errors_csv)

