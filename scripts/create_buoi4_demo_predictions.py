from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEMO_ROWS = [
    {
        "image_id": "demo_001",
        "gt": "51H12345",
        "pred_a": "51H12345",
        "pred_b": "51H12345",
        "latency_a": 82.4,
        "latency_b": 138.7,
        "bbox": "120,80,310,140",
    },
    {
        "image_id": "demo_002",
        "gt": "30A56789",
        "pred_a": "30A5678",
        "pred_b": "30A56789",
        "latency_a": 79.1,
        "latency_b": 141.2,
        "bbox": "90,60,260,118",
    },
    {
        "image_id": "demo_003",
        "gt": "59G11234",
        "pred_a": "59G11234",
        "pred_b": "59G11234",
        "latency_a": 85.8,
        "latency_b": 149.9,
        "bbox": "132,95,322,155",
    },
    {
        "image_id": "demo_004",
        "gt": "43B67890",
        "pred_a": "43B6789O",
        "pred_b": "43B67890",
        "latency_a": 88.6,
        "latency_b": 152.5,
        "bbox": "110,70,300,132",
    },
    {
        "image_id": "demo_005",
        "gt": "29C24680",
        "pred_a": "29C24680",
        "pred_b": "29C2468O",
        "latency_a": 81.3,
        "latency_b": 143.4,
        "bbox": "100,88,286,147",
    },
    {
        "image_id": "demo_006",
        "gt": "61D13579",
        "pred_a": "61D13579",
        "pred_b": "61D13579",
        "latency_a": 84.9,
        "latency_b": 146.8,
        "bbox": "78,55,248,112",
    },
    {
        "image_id": "demo_007",
        "gt": "50F99881",
        "pred_a": "50F9981",
        "pred_b": "50F99881",
        "latency_a": 87.2,
        "latency_b": 155.6,
        "bbox": "150,100,345,164",
    },
    {
        "image_id": "demo_008",
        "gt": "72A45678",
        "pred_a": "72A45678",
        "pred_b": "72A45678",
        "latency_a": 80.7,
        "latency_b": 139.3,
        "bbox": "130,92,315,150",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create small Buoi 4 demo prediction CSVs for A/B metric smoke tests."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/buoi4/demo"))
    return parser.parse_args()


def _error_type(gt: str, pred: str) -> str:
    return "ok" if gt == pred else "ocr_or_spotting"


def _write_prediction_csv(path: Path, variant: str) -> None:
    pred_key = "pred_a" if variant == "a" else "pred_b"
    latency_key = "latency_a" if variant == "a" else "latency_b"

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["image_id", "gt", "pred", "score", "latency_ms", "bbox_xyxy", "error_type"],
        )
        writer.writeheader()
        for idx, row in enumerate(DEMO_ROWS):
            pred = str(row[pred_key])
            score = 0.92 - idx * 0.015 if variant == "a" else 0.89 - idx * 0.01
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "gt": row["gt"],
                    "pred": pred,
                    "score": f"{score:.3f}",
                    "latency_ms": row[latency_key],
                    "bbox_xyxy": row["bbox"],
                    "error_type": _error_type(str(row["gt"]), pred),
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_a_csv = args.output_dir / "deepsolo_e2e_predictions.csv"
    config_b_csv = args.output_dir / "deepsolo_trocr_predictions.csv"
    _write_prediction_csv(config_a_csv, "a")
    _write_prediction_csv(config_b_csv, "b")

    summary = {
        "note": "Demo CSV dùng để kiểm tra code metric, chưa phải kết quả mô hình thật.",
        "config_a_csv": str(config_a_csv),
        "config_b_csv": str(config_b_csv),
        "num_samples": len(DEMO_ROWS),
    }
    (args.output_dir / "README.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
