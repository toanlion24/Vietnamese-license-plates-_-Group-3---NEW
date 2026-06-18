"""
Eval Qwen2-VL base (no LoRA) OCR on 550 crops.
LoRA not compatible with full base model — using base only.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))

import csv, json, time, logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from src.postprocess.plate_rules import normalize_plate_text

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = _Path(__file__).parent.parent


SYSTEM_PROMPT = (
    "Ban la he thong nhan dien bien so xe Viet Nam. "
    "Doc va tra loi chi bien so xe, khong giai thich. "
    "Dinh dang: [ma tinh][chu cai loai][so]. Vi du: 30G112345"
)


@dataclass
class OcrResult:
    image_id: str
    text_raw: str
    text_norm: str
    ocr_score: float = 1.0


class Qwen2VLEval:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct",
                 device="cuda", max_new_tokens=32, temperature=0.0):
        logger.info("Loading Qwen2-VL base model: %s", model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = Qwen2VLProcessor.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        logger.info("Model ready. VRAM: %.2f GB", torch.cuda.memory_allocated() / 1e9)

    def _preprocess(self, crop_np: np.ndarray) -> Image.Image:
        if len(crop_np.shape) == 2:
            pil_img = Image.fromarray(crop_np, mode="L").convert("RGB")
        else:
            pil_img = Image.fromarray(crop_np)
        w, h = pil_img.size
        target = 448
        if max(w, h) > target:
            ratio = target / max(w, h)
            pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return pil_img

    def recognize(self, image_id: str, crop_np: np.ndarray) -> OcrResult:
        pil_img = self._preprocess(crop_np)

        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": "Doc bien so xe trong anh nay:"},
            ]},
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(
            text=[text], images=image_inputs,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.processor.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        text_norm = normalize_plate_text(response)

        return OcrResult(
            image_id=image_id,
            text_raw=response,
            text_norm=text_norm,
        )


def levenshtein(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def cer(records):
    total_dist, total_chars = 0, 0
    for r in records:
        gt = normalize_plate_text(r["gt"])
        pred = normalize_plate_text(r["pred"])
        total_dist += levenshtein(gt, pred)
        total_chars += max(1, len(gt))
    return total_dist / total_chars if total_chars else 0.0


def wer(records):
    total_dist, total_words = 0, 0
    for r in records:
        gt = normalize_plate_text(r["gt"])
        pred = normalize_plate_text(r["pred"])
        total_dist += levenshtein(gt, pred)
        total_words += max(1, len(gt))
    return total_dist / total_words if total_words else 0.0


def plate_accuracy(records):
    if not records: return 0.0
    return sum(
        normalize_plate_text(r["gt"]) == normalize_plate_text(r["pred"])
        for r in records
    ) / len(records)


def classify_error(pred, gt):
    pred_norm = normalize_plate_text(pred)
    gt_norm = normalize_plate_text(gt)
    if not pred_norm: return "empty"
    if pred_norm == gt_norm: return "none"
    pred_chars = set(pred_norm)
    gt_chars = set(gt_norm)
    if pred_chars and gt_chars and not pred_chars.issubset(gt_chars.union({"O", "0", "I", "l", "1"})):
        return "hallucination"
    if len(pred_norm) < len(gt_norm) * 0.5: return "truncation"
    if len(pred_norm) > len(gt_norm) * 1.5: return "overreading"
    return "substitution"


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=_Path,
                        default=PROJECT_ROOT / "data" / "labels_manual.csv")
    parser.add_argument("--crops-dir", type=_Path,
                        default=PROJECT_ROOT / "data" / "crops")
    parser.add_argument("--output-dir", type=_Path,
                        default=PROJECT_ROOT / "outputs" / "qwen2vl_base_eval")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.read_text(encoding="utf-8").strip().splitlines()))
    logger.info("Loaded %d manifest entries", len(manifest))

    ocr = Qwen2VLEval(model_name=args.model, device=args.device)
    samples = manifest[:args.max_samples]

    records = []
    timings = []
    logger.info("Running OCR on %d samples...", len(samples))

    for i, row in enumerate(samples):
        img_path = args.crops_dir / f"{row['image_id']}.jpg"
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.perf_counter()
        result = ocr.recognize(row["image_id"], frame)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed)

        error = classify_error(result.text_raw, row["text_gt"])
        records.append({
            "image_id": row["image_id"],
            "gt": row["text_gt"],
            "pred": result.text_raw,
            "error_type": error,
        })

        if (i + 1) % 50 == 0:
            acc = plate_accuracy(records)
            logger.info("  %d/%d | Acc=%.1f%% | CER=%.4f | Lat=%.1fms",
                       i + 1, len(samples), acc * 100, cer(records),
                       sum(timings) / len(timings))

    # Save CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "gt", "pred", "error_type"])
        writer.writeheader()
        writer.writerows(records)

    cer_val = cer(records)
    wer_val = wer(records)
    acc_val = plate_accuracy(records)
    error_counts = {}
    for r in records:
        if r["error_type"] != "none":
            error_counts[r["error_type"]] = error_counts.get(r["error_type"], 0) + 1

    metrics = {
        "num_samples": len(records),
        "cer": round(cer_val, 4),
        "wer": round(wer_val, 4),
        "plate_accuracy": round(acc_val, 4),
        "mean_latency_ms": round(sum(timings) / len(timings), 2) if timings else 0,
        "total_latency_s": round(sum(timings) / 1000, 2),
        "num_correct": sum(
            1 for r in records
            if normalize_plate_text(r["gt"]) == normalize_plate_text(r["pred"])
        ),
        "error_breakdown": error_counts,
        "predictions_csv": str(csv_path),
    }

    json_path = args.output_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS: Accuracy=%.1f%%  CER=%.4f  WER=%.4f  Latency=%.1fms/img",
                acc_val * 100, cer_val, wer_val,
                metrics["mean_latency_ms"])
    logger.info("Error breakdown: %s", error_counts)
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 60)

    # Show sample errors
    errors = [r for r in records if r["error_type"] not in ("none",)]
    logger.info("\nSample errors (first 10):")
    for r in errors[:10]:
        logger.info("  %s | GT=%s | PRED=%s | %s", r["image_id"], r["gt"], r["pred"], r["error_type"])

    return metrics


if __name__ == "__main__":
    run()
