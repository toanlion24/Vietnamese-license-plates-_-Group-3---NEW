from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics_plate import cer, plate_accuracy, wer
from src.postprocess.plate_rules import normalize_plate_text
from src.utils.types import EvalRecord


@dataclass(slots=True)
class PredictionSummary:
    name: str
    csv_path: Path
    records: list[EvalRecord]
    image_ids: set[str]
    latencies_ms: list[float]
    error_counts: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Buoi 4 DeepSolo end-to-end vs DeepSolo + TrOCR predictions."
    )
    parser.add_argument(
        "--config-a-csv",
        type=Path,
        required=True,
        help="CSV for config A: DeepSolo end-to-end. Required columns: image_id, gt, pred.",
    )
    parser.add_argument(
        "--config-b-csv",
        type=Path,
        required=True,
        help="CSV for config B: DeepSolo + TrOCR. Required columns: image_id, gt, pred.",
    )
    parser.add_argument("--metrics-json", type=Path, default=Path("reports/buoi4_ab_metrics.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/ablation_deepsolo_trocr.md"))
    parser.add_argument(
        "--experiment-note",
        type=str,
        default="",
        help="Optional note rendered near the top of the Markdown report.",
    )
    return parser.parse_args()


def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value:
            return value
    return ""


def _parse_latency(row: dict[str, str]) -> float | None:
    raw = row.get("latency_ms", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_prediction_csv(path: Path, name: str) -> PredictionSummary:
    records: list[EvalRecord] = []
    image_ids: set[str] = set()
    latencies_ms: list[float] = []
    error_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader, start=1):
            image_id = _first_non_empty(row, ("image_id", "id", "filename", "file_name"))
            if not image_id:
                image_id = f"row_{idx}"
            gt = _first_non_empty(row, ("gt", "text_gt", "label", "target"))
            pred = _first_non_empty(row, ("pred", "text_pred", "plate_text", "prediction"))
            error_type = row.get("error_type", "")
            if not error_type:
                error_type = "ok" if normalize_plate_text(gt) == normalize_plate_text(pred) else "ocr_or_spotting"

            latency = _parse_latency(row)
            if latency is not None:
                latencies_ms.append(latency)

            image_ids.add(image_id)
            error_counts[error_type] += 1
            records.append(EvalRecord(image_id=image_id, gt=gt, pred=pred, error_type=error_type))

    return PredictionSummary(
        name=name,
        csv_path=path,
        records=records,
        image_ids=image_ids,
        latencies_ms=latencies_ms,
        error_counts=error_counts,
    )


def summarize(summary: PredictionSummary) -> dict[str, object]:
    mean_latency = None
    if summary.latencies_ms:
        mean_latency = sum(summary.latencies_ms) / len(summary.latencies_ms)

    return {
        "csv_path": str(summary.csv_path),
        "num_samples": len(summary.records),
        "cer": cer(summary.records),
        "wer": wer(summary.records),
        "plate_accuracy": plate_accuracy(summary.records),
        "mean_latency_ms": mean_latency,
        "error_counts": dict(summary.error_counts),
    }


def choose_pipeline(metrics_a: dict[str, object], metrics_b: dict[str, object]) -> str:
    acc_a = float(metrics_a["plate_accuracy"])
    acc_b = float(metrics_b["plate_accuracy"])
    cer_a = float(metrics_a["cer"])
    cer_b = float(metrics_b["cer"])

    if acc_b > acc_a:
        return "Tạm chọn cấu hình B (DeepSolo + TrOCR) vì plate accuracy cao hơn."
    if acc_a > acc_b:
        return "Tạm chọn cấu hình A (DeepSolo end-to-end) vì plate accuracy cao hơn."
    if cer_b < cer_a:
        return "Hai cấu hình có plate accuracy bằng nhau; tạm chọn B vì CER thấp hơn."
    if cer_a < cer_b:
        return "Hai cấu hình có plate accuracy bằng nhau; tạm chọn A vì CER thấp hơn."
    return "Hai cấu hình đang ngang nhau theo metric chính; chọn cấu hình dễ debug hơn cho Buổi 5."


def _format_float(value: object) -> str:
    if value is None:
        return "không có"
    return f"{float(value):.4f}"


def render_report(
    config_a: PredictionSummary,
    config_b: PredictionSummary,
    metrics_a: dict[str, object],
    metrics_b: dict[str, object],
    warnings: list[str],
    experiment_note: str = "",
) -> str:
    recommendation = choose_pipeline(metrics_a, metrics_b)
    generated_at = datetime.now(timezone.utc).isoformat()

    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- Không có cảnh báo."
    error_a = "\n".join(f"- {key}: {value}" for key, value in config_a.error_counts.items()) or "- Không có."
    error_b = "\n".join(f"- {key}: {value}" for key, value in config_b.error_counts.items()) or "- Không có."
    note_section = f"\nGhi chú: {experiment_note}\n" if experiment_note else ""

    return f"""# Báo cáo A/B Buổi 4 - DeepSolo end-to-end vs DeepSolo + TrOCR

Tự động cập nhật từ `scripts/run_buoi4_experiments.py`.
{note_section}

## 1) Thiết lập

- Thời điểm tạo báo cáo: `{generated_at}`
- Cấu hình A: DeepSolo end-to-end
- File A: `{config_a.csv_path}`
- Cấu hình B: DeepSolo + TrOCR
- File B: `{config_b.csv_path}`

## 2) Kiểm tra công bằng

{warning_lines}

## 3) Kết quả định lượng

Cấu hình A - DeepSolo end-to-end:

- Số mẫu: {metrics_a["num_samples"]}
- CER: {_format_float(metrics_a["cer"])}
- WER: {_format_float(metrics_a["wer"])}
- Plate accuracy: {_format_float(metrics_a["plate_accuracy"])}
- Mean latency ms: {_format_float(metrics_a["mean_latency_ms"])}

Cấu hình B - DeepSolo + TrOCR:

- Số mẫu: {metrics_b["num_samples"]}
- CER: {_format_float(metrics_b["cer"])}
- WER: {_format_float(metrics_b["wer"])}
- Plate accuracy: {_format_float(metrics_b["plate_accuracy"])}
- Mean latency ms: {_format_float(metrics_b["mean_latency_ms"])}

## 4) Phân bố lỗi

Cấu hình A:

{error_a}

Cấu hình B:

{error_b}

## 5) Quyết định tạm thời

{recommendation}

## 6) Việc cần làm tiếp

- Mở các case sai và gán lại loại lỗi: `detect_miss`, `bad_crop`, `ocr_error`, `postprocess_helped`, `ambiguous_gt`.
- Chọn 10-20 hard cases để đưa vào báo cáo cuối.
- Nếu chọn cấu hình B, ưu tiên cải thiện crop/rectify trước khi fine-tune TrOCR.
- Nếu chọn cấu hình A, kiểm tra riêng lỗi spotting sai vùng và lỗi nhận dạng sai text.
"""


def main() -> None:
    args = parse_args()
    config_a = load_prediction_csv(args.config_a_csv, "deepsolo_e2e")
    config_b = load_prediction_csv(args.config_b_csv, "deepsolo_trocr")

    metrics_a = summarize(config_a)
    metrics_b = summarize(config_b)
    common_ids = config_a.image_ids & config_b.image_ids
    warnings: list[str] = []
    if config_a.image_ids != config_b.image_ids:
        warnings.append(
            "Hai CSV không có cùng tập `image_id`; hãy kiểm tra lại để so sánh công bằng."
        )
        warnings.append(f"Số ảnh chung: {len(common_ids)}")
        warnings.append(f"Chỉ có trong A: {len(config_a.image_ids - config_b.image_ids)}")
        warnings.append(f"Chỉ có trong B: {len(config_b.image_ids - config_a.image_ids)}")
    if not config_a.records:
        warnings.append("CSV cấu hình A không có record.")
    if not config_b.records:
        warnings.append("CSV cấu hình B không có record.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_a": metrics_a,
        "config_b": metrics_b,
        "fairness": {
            "same_image_ids": config_a.image_ids == config_b.image_ids,
            "common_image_count": len(common_ids),
            "warnings": warnings,
        },
        "recommendation": choose_pipeline(metrics_a, metrics_b),
    }

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        render_report(config_a, config_b, metrics_a, metrics_b, warnings, args.experiment_note),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
