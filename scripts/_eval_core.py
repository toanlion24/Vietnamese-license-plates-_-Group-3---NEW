"""Actual evaluation core — runs as __main__ or is imported."""
import gc, os, sys, csv, json, time, logging
from pathlib import Path

os.environ.setdefault("ARROW_DISABLE_MMAP", "1")
os.environ.setdefault("PYARROW_CSV_IPC_ENABLE", "0")

PROJECT_ROOT = Path(__file__).parent.parent if "__file__" in globals() else Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from src.postprocess.plate_rules import normalize_plate_text

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class OcrEvaluator:
    def __init__(self, model_name, lora_path=None, device="cuda",
                 max_new_tokens=32, temperature=0.0):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info("Loading: %s", model_name)
        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map="cuda:0", low_cpu_mem_usage=True,
        )
        if lora_path:
            logger.info("Applying LoRA: %s", lora_path)
            self.model = PeftModel.from_pretrained(self.model, str(lora_path))
            self.model = self.model.merge_and_unload()
            logger.info("LoRA merged")
        self.model.eval()
        self.processor = Qwen2VLProcessor.from_pretrained(model_name)
        logger.info("Ready in %.0fs. VRAM: %.2f GB",
                    time.time() - t0, torch.cuda.memory_allocated() / 1e9)

    def _preprocess(self, crop_np):
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

    def recognize(self, image_id, crop_np):
        pil_img = self._preprocess(crop_np)
        conv = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": "Doc bien so xe trong anh nay:"},
        ]}]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conv)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature, do_sample=self.temperature > 0,
            )
        input_len = inputs["input_ids"].shape[1]
        return normalize_plate_text(
            self.processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
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


def classify_error(pred_raw, gt):
    pred = normalize_plate_text(pred_raw)
    gt_norm = normalize_plate_text(gt)
    if not pred: return "empty"
    if pred == gt_norm: return "none"
    pc, gc_set = set(pred), set(gt_norm)
    if pc and gc_set and not pc.issubset(gc_set.union({"O", "0", "I", "l", "1"})):
        return "hallucination"
    if len(pred) < len(gt_norm) * 0.5: return "truncation"
    if len(pred) > len(gt_norm) * 1.5: return "overreading"
    return "substitution"


def compute_metrics(records):
    total_dist = total_chars = 0
    for r in records:
        gt = normalize_plate_text(r["gt"])
        pred = normalize_plate_text(r["pred"])
        total_dist += levenshtein(gt, pred)
        total_chars += max(1, len(gt))
    cer = total_dist / total_chars if total_chars else 0.0
    acc = sum(normalize_plate_text(r["gt"]) == normalize_plate_text(r["pred"]) for r in records) / len(records) if records else 0.0
    err_counts = {}
    for r in records:
        e = classify_error(r["pred"], r["gt"])
        err_counts[e] = err_counts.get(e, 0) + 1
    return {
        "cer": round(cer, 4), "accuracy": round(acc, 4),
        "num_samples": len(records),
        "num_correct": sum(1 for r in records if normalize_plate_text(r["gt"]) == normalize_plate_text(r["pred"])),
        "error_breakdown": err_counts,
    }


def run_eval(name, evaluator, samples, crops_dir):
    logger.info("=" * 50 + "\nEVAL: %s\n" + "=" * 50, name)
    records, timings = [], []
    for i, row in enumerate(samples):
        img_path = crops_dir / f"{row['image_id']}.jpg"
        if not img_path.exists():
            img_path = crops_dir / f"{row['image_id']}.png"
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        t0 = time.perf_counter()
        pred = evaluator.recognize(row["image_id"], frame)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed)
        records.append({"image_id": row["image_id"], "gt": row["text_gt"],
                        "pred": pred, "error_type": classify_error(pred, row["text_gt"])})
        if (i + 1) % 50 == 0:
            m = compute_metrics(records)
            logger.info("%d/%d | Acc=%.1f%% CER=%.4f Lat=%.1fms",
                       i + 1, len(samples), m["accuracy"]*100, m["cer"], sum(timings)/len(timings))
    metrics = compute_metrics(records)
    metrics["mean_latency_ms"] = round(sum(timings)/len(timings), 2)
    logger.info("RESULT: Acc=%.1f%% CER=%.4f Lat=%.1fms",
                metrics["accuracy"]*100, metrics["cer"], metrics["mean_latency_ms"])
    errors = [r for r in records if r["error_type"] not in ("none",)]
    logger.info("Sample errors:")
    for r in errors[:10]:
        logger.info("  %s | GT=%s | PRED=%s | %s", r["image_id"], r["gt"], r["pred"], r["error_type"])
    return records, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--lora-path", type=Path, default=PROJECT_ROOT / "experiments" / "qwen2vl_crops_lora")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "crops" / "manifest.csv")
    parser.add_argument("--crops-dir", type=Path, default=PROJECT_ROOT / "data" / "crops")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "lora_comparison")
    parser.add_argument("--max-samples", type=int, default=550)
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.read_text(encoding="utf-8").strip().splitlines()))
    samples = manifest[:args.max_samples]
    logger.info("Manifest: %d | Evaluating: %d", len(manifest), len(samples))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # BASE
    base_eval = OcrEvaluator(model_name=args.base_model, lora_path=None)
    base_rec, base_m = run_eval("BASE", base_eval, samples, args.crops_dir)
    del base_eval; gc.collect(); torch.cuda.empty_cache()

    # FINE-TUNED
    lora_eval = OcrEvaluator(model_name=args.base_model, lora_path=args.lora_path)
    lora_rec, lora_m = run_eval("FINE-TUNED", lora_eval, samples, args.crops_dir)
    del lora_eval; gc.collect(); torch.cuda.empty_cache()

    logger.info("\n" + "=" * 50 + "\nCOMPARISON SUMMARY\n" + "=" * 50)
    logger.info("                   BASE        FINE-TUNED    DELTA")
    logger.info("Accuracy:       %.2f%%       %.2f%%        %+.2f%%",
                base_m["accuracy"]*100, lora_m["accuracy"]*100,
                (lora_m["accuracy"]-base_m["accuracy"])*100)
    logger.info("CER:           %.4f       %.4f        %+.4f",
                base_m["cer"], lora_m["cer"], lora_m["cer"]-base_m["cer"])
    logger.info("Correct:       %d/%d        %d/%d",
                base_m["num_correct"], base_m["num_samples"],
                lora_m["num_correct"], lora_m["num_samples"])

    comparison = {
        "base": base_m, "lora": lora_m,
        "improvement": {
            "accuracy_delta": round(lora_m["accuracy"]-base_m["accuracy"], 4),
            "cer_delta": round(lora_m["cer"]-base_m["cer"], 4),
        }
    }

    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "predictions_side_by_side.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id","gt","pred_base","pred_lora","correct_base","correct_lora"])
        writer.writeheader()
        for br, lr in zip(base_rec, lora_rec):
            writer.writerow({
                "image_id": br["image_id"], "gt": br["gt"],
                "pred_base": br["pred"], "pred_lora": lr["pred"],
                "correct_base": "Y" if br["error_type"]=="none" else "N",
                "correct_lora": "Y" if lr["error_type"]=="none" else "N",
            })

    for label, recs in [("base", base_rec), ("lora", lora_rec)]:
        with (args.output_dir / f"predictions_{label}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_id","gt","pred","error_type"])
            w.writeheader(); w.writerows(recs)

    logger.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
