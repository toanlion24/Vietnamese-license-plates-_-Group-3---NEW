"""
Evaluation script for VN license plate OCR pipeline.

Usage:
    # Evaluate Qwen2VL fine-tuned
    python -m src.eval.eval_ocr_pipeline --ocr-backend qwen2vl

    # Evaluate EasyOCR
    python -m src.eval.eval_ocr_pipeline --ocr-backend easyocr

    # Compare multiple backends
    python -m src.eval.eval_ocr_pipeline --ocr-backend qwen2vl easyocr trocr
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from src.ocr.base import DummyOcr
from src.ocr.qwen_adapter import Qwen2VLPlateOcr
from src.postprocess.plate_rules import normalize_plate_text
from src.utils.types import EvalRecord, PlateCrop

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _build_ocr(backend: str, device: str = "cpu") -> "PlateOcr":
    """Build OCR backend by name."""
    if backend == "dummy":
        return DummyOcr()
    if backend == "qwen2vl":
        return Qwen2VLPlateOcr(
            model_name=str(PROJECT_ROOT / "experiments" / "qwen2vl_finetuned"),
            device=device,
            cache_dir=str(PROJECT_ROOT / ".cache" / "huggingface"),
            max_new_tokens=32,
            temperature=0.1,
        )
    raise ValueError(f"Unknown backend: {backend}")


def levenshtein(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def cer(records: list[EvalRecord]) -> float:
    total_dist, total_chars = 0, 0
    for r in records:
        gt = normalize_plate_text(r.gt)
        pred = normalize_plate_text(r.pred)
        total_dist += levenshtein(gt, pred)
        total_chars += max(1, len(gt))
    return total_dist / total_chars if total_chars else 0.0


def wer(records: list[EvalRecord]) -> float:
    total_dist, total_words = 0, 0
    for r in records:
        gt = [normalize_plate_text(p) for p in r.gt.replace("\n", " ").split() if normalize_plate_text(p)]
        pred = [normalize_plate_text(p) for p in r.pred.replace("\n", " ").split() if normalize_plate_text(p)]
        if not gt:
            gt = [normalize_plate_text(r.gt)]
        if not pred:
            pred = [normalize_plate_text(r.pred)]
        total_dist += levenshtein("".join(gt), "".join(pred))
        total_words += max(1, len(gt))
    return total_dist / total_words if total_words else 0.0


def plate_accuracy(records: list[EvalRecord]) -> float:
    if not records:
        return 0.0
    correct = sum(normalize_plate_text(r.gt) == normalize_plate_text(r.pred) for r in records)
    return correct / len(records)


def classify_error(pred: str, gt: str) -> str:
    """Classify the type of recognition error."""
    pred_norm = normalize_plate_text(pred)
    gt_norm = normalize_plate_text(gt)
    if not pred_norm:
        return "empty"
    if pred_norm == gt_norm:
        return "none"
    pred_chars = set(pred_norm)
    gt_chars = set(gt_norm)
    if pred_chars and gt_chars and not pred_chars.issubset(gt_chars.union({"O", "0", "I", "l", "1"})):
        return "hallucination"
    if len(pred_norm) < len(gt_norm) * 0.5:
        return "truncation"
    if len(pred_norm) > len(gt_norm) * 1.5:
        return "overreading"
    return "substitution"



def load_manifest(csv_path: Path) -> list[dict]:
    """Load manifest CSV with image_id, text_gt."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def iter_crops(manifest: list[dict], crops_dir: Path, extensions=("jpg", "jpeg", "png")):
    """Yield PlateCrop for each manifest entry."""
    for row in manifest:
        img_id = row["image_id"]
        gt = row.get("text_gt", "")
        for ext in extensions:
            img_path = crops_dir / f"{img_id}.{ext}"
            if img_path.exists():
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    yield PlateCrop(
                        image_id=img_id,
                        crop=frame,
                        bbox_xyxy=(0, 0, frame.shape[1], frame.shape[0]),
                        det_score=1.0,
                    ), gt
                break


def run_evaluation(
    ocr,
    manifest: list[dict],
    crops_dir: Path,
    output_csv: Path,
    output_json: Path,
    max_samples: int | None = None,
) -> dict:
    """Run OCR evaluation on manifest."""
    records: list[EvalRecord] = []
    timings: list[float] = []

    samples = list(iter_crops(manifest, crops_dir))
    if max_samples:
        samples = samples[:max_samples]

    logger.info("Evaluating %d samples...", len(samples))

    for i, (plate_crop, gt) in enumerate(samples):
        t0 = time.perf_counter()
        result = ocr.recognize(plate_crop, preprocessed=None)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed)

        pred = normalize_plate_text(result.text_raw)
        error_type = classify_error(pred, gt)

        records.append(EvalRecord(
            image_id=plate_crop.image_id,
            gt=gt,
            pred=result.text_raw,
            error_type=error_type,
        ))

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d samples...", i + 1, len(samples))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "gt", "pred", "error_type"])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    metrics = {
        "num_samples": len(records),
        "cer": round(cer(records), 4),
        "wer": round(wer(records), 4),
        "plate_accuracy": round(plate_accuracy(records), 4),
        "mean_latency_ms": round(sum(timings) / len(timings), 2) if timings else 0,
        "total_latency_s": round(sum(timings) / 1000, 2),
        "num_correct": sum(1 for r in records if normalize_plate_text(r.gt) == normalize_plate_text(r.pred)),
        "predictions_csv": str(output_csv),
    }

    # Error analysis
    error_counts: dict[str, int] = {}
    for r in records:
        if r.error_type != "none":
            error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1

    metrics["error_breakdown"] = error_counts

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VN plate OCR pipeline")
    parser.add_argument(
        "--ocr-backend",
        nargs="+",
        default=["qwen2vl"],
        choices=["qwen2vl", "dummy"],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "labels_manual.csv",
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "crops",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "qwen2vl_eval",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    logger.info("Loaded %d manifest entries from %s", len(manifest), args.manifest)

    results: dict[str, dict] = {}

    for backend in args.ocr_backend:
        logger.info("=" * 60)
        logger.info("Evaluating backend: %s", backend)
        logger.info("=" * 60)

        output_csv = args.output_dir / f"{backend}_predictions.csv"
        output_json = args.output_dir / f"{backend}_metrics.json"

        ocr = _build_ocr(backend, device=args.device)

        metrics = run_evaluation(
            ocr=ocr,
            manifest=manifest,
            crops_dir=args.crops_dir,
            output_csv=output_csv,
            output_json=output_json,
            max_samples=args.max_samples,
        )

        results[backend] = metrics

        logger.info(
            "  Accuracy: %.1f%%  CER: %.4f  WER: %.4f  Latency: %.1fms/img",
            metrics["plate_accuracy"] * 100,
            metrics["cer"],
            metrics["wer"],
            metrics["mean_latency_ms"],
        )
        logger.info("  Error breakdown: %s", metrics["error_breakdown"])

    # Summary comparison
    if len(results) > 1:
        logger.info("=" * 60)
        logger.info("SUMMARY COMPARISON")
        logger.info("=" * 60)
        print(f"\n{'Backend':<20} {'Accuracy':>10} {'CER':>8} {'WER':>8} {'Latency':>12}")
        print("-" * 62)
        for name, m in results.items():
            print(f"{name:<20} {m['plate_accuracy']*100:>9.1f}% {m['cer']:>8.4f} {m['wer']:>8.4f} {m['mean_latency_ms']:>10.1f}ms")
        print()

        # Save comparison
        summary_path = args.output_dir / "comparison.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Comparison saved to %s", summary_path)


if __name__ == "__main__":
    main()
