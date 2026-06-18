"""Compare Qwen2-VL base vs fine-tuned LoRA OCR on 550 VN plate crops.
sys.path is sanitized of ALL ComputerVisionNew paths — project modules are NOT in sys.path
at import time, but since project imports are via absolute paths (installed in site-packages
or accessed via absolute Path references), no path-scanning of project directory occurs.
"""
import sys as _sys
from pathlib import Path

# ── MUST be the first lines: remove ALL ComputerVisionNew paths ───────────────────────────
for _p in list(_sys.path):
    if "ComputerVisionNew" in _p or "computerVisionNew" in _p:
        _sys.path.remove(_p)
# Do NOT re-add d:/ComputerVisionNew/src — the stale kernel transaction
# causes pyarrow/sklearn import to fail whenever ANY path in sys.path contains the project dir.

_PROJECT_ROOT = Path("d:/ComputerVisionNew")

def _p(msg=""):
    print(msg, flush=True)

# ── All imports after path sanitization ──────────────────────────────────────────
import gc, csv, json, time, re
import numpy, torch, cv2
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

_LORA_DIR = _PROJECT_ROOT / "experiments" / "qwen2vl_crops_lora"
_BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def _normalize(text):
    if not text:
        return ""
    for old, new in [("O", "0"), ("o", "0"), ("I", "1"), ("l", "1"), (" ", "")]:
        text = text.replace(old, new)
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


class _OcrEvaluator:
    def __init__(self, model_name=_BASE_MODEL, lora_path=None, label=""):
        _p(f"[{label}] Loading: {model_name}")
        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map="cuda:0", low_cpu_mem_usage=True,
        )
        if lora_path:
            _p(f"[{label}] Applying LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, str(lora_path))
            _p(f"[{label}] LoRA ready (PEFT mode, no merge)")
        self.model.eval()
        self.processor = Qwen2VLProcessor.from_pretrained(model_name)
        _p(f"[{label}] Ready in {time.time()-t0:.0f}s. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def _preprocess(self, crop_np):
        img = Image.fromarray(crop_np)
        w, h = img.size
        target = 448
        if max(w, h) > target:
            ratio = target / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    def recognize(self, image_id, crop_np):
        pil_img = self._preprocess(crop_np)
        conv = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": "Doc bien so xe trong anh nay:"},
        ]}]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conv)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        return _normalize(
            self.processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        )


def _levenshtein(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _classify_error(pred_raw, gt):
    pred = _normalize(pred_raw)
    gt_norm = _normalize(gt)
    if not pred: return "empty"
    if pred == gt_norm: return "none"
    pc, gc_set = set(pred), set(gt_norm)
    if pc and gc_set and not pc.issubset(gc_set.union({"O", "0", "I", "l", "1"})):
        return "hallucination"
    if len(pred) < len(gt_norm) * 0.5: return "truncation"
    if len(pred) > len(gt_norm) * 1.5: return "overreading"
    return "substitution"


def _compute_metrics(records):
    total_dist = total_chars = 0
    for r in records:
        gt = _normalize(r["gt"])
        pred = _normalize(r["pred"])
        total_dist += _levenshtein(gt, pred)
        total_chars += max(1, len(gt))
    cer = total_dist / total_chars if total_chars else 0.0
    acc = sum(_normalize(r["gt"]) == _normalize(r["pred"]) for r in records) / len(records) if records else 0.0
    err_counts = {}
    for r in records:
        e = _classify_error(r["pred"], r["gt"])
        err_counts[e] = err_counts.get(e, 0) + 1
    return {
        "cer": round(cer, 4), "accuracy": round(acc, 4),
        "num_samples": len(records),
        "num_correct": sum(1 for r in records if _normalize(r["gt"]) == _normalize(r["pred"])),
        "error_breakdown": err_counts,
    }


def _run_eval(name, evaluator, samples, crops_dir):
    _p(f"\n=== EVAL: {name} ===")
    records, timings = [], []
    for i, row in enumerate(samples):
        img_path = crops_dir / (row["image_id"] + ".jpg")
        if not img_path.exists():
            img_path = crops_dir / (row["image_id"] + ".png")
        frame = cv2.imread(str(img_path))
        if frame is None:
            _p(f"  WARNING: cannot read {img_path}")
            continue
        t0 = time.perf_counter()
        try:
            pred = evaluator.recognize(row["image_id"], frame)
        except Exception as ex:
            _p(f"  WARNING: {row['image_id']} error: {ex}")
            pred = ""
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed)
        records.append({
            "image_id": row["image_id"], "gt": row["text_gt"],
            "pred": pred, "error_type": _classify_error(pred, row["text_gt"]),
        })
        if (i + 1) % 50 == 0:
            m = _compute_metrics(records)
            _p(f"  {i+1}/{len(samples)} | Acc={m['accuracy']*100:.1f}% | CER={m['cer']:.4f} | Lat={sum(timings)/len(timings):.1f}ms")
    metrics = _compute_metrics(records)
    metrics["mean_latency_ms"] = round(sum(timings)/len(timings), 2)
    _p(f"RESULT: Acc={metrics['accuracy']*100:.1f}% CER={metrics['cer']:.4f} Lat={metrics['mean_latency_ms']:.1f}ms/img")
    errors = [r for r in records if r["error_type"] not in ("none",)]
    _p("Sample errors:")
    for r in errors[:10]:
        _p(f"  {r['image_id']} | GT={r['gt']} | PRED={r['pred']} | {r['error_type']}")
    return records, metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=_PROJECT_ROOT / "data" / "crops" / "manifest.csv")
    parser.add_argument("--crops-dir", type=Path, default=_PROJECT_ROOT / "data" / "crops")
    parser.add_argument("--lora-path", type=Path, default=_LORA_DIR)
    parser.add_argument("--base-model", default=_BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "outputs" / "lora_comparison")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    manifest = list(csv.DictReader(args.manifest.read_text(encoding="utf-8").strip().splitlines()))
    samples = manifest[:args.max_samples]
    _p(f"Manifest: {len(manifest)} entries | Evaluating: {len(samples)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _p("\n" + "=" * 55)
    _p("STEP 1: BASE MODEL (no fine-tuning)")
    _p("=" * 55)
    base_eval = _OcrEvaluator(model_name=args.base_model, lora_path=None, label="BASE")
    base_rec, base_m = _run_eval("BASE MODEL", base_eval, samples, args.crops_dir)
    del base_eval; gc.collect(); torch.cuda.empty_cache()

    _p("\n" + "=" * 55)
    _p("STEP 2: FINE-TUNED LoRA MODEL")
    _p("=" * 55)
    lora_eval = _OcrEvaluator(model_name=args.base_model, lora_path=args.lora_path, label="LORA")
    lora_rec, lora_m = _run_eval("FINE-TUNED LoRA", lora_eval, samples, args.crops_dir)
    del lora_eval; gc.collect(); torch.cuda.empty_cache()

    _p("\n" + "=" * 55)
    _p("COMPARISON SUMMARY")
    _p("=" * 55)
    _p("                   BASE        FINE-TUNED    DELTA")
    _p(f"Accuracy:       {base_m['accuracy']*100:.2f}%       {lora_m['accuracy']*100:.2f}%        {(lora_m['accuracy']-base_m['accuracy'])*100:+.2f}%")
    _p(f"CER:           {base_m['cer']:.4f}       {lora_m['cer']:.4f}        {lora_m['cer']-base_m['cer']:+.4f}")
    _p(f"Correct:       {base_m['num_correct']}/{base_m['num_samples']}        {lora_m['num_correct']}/{lora_m['num_samples']}")
    _p("=" * 55)

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
        w = csv.DictWriter(f, fieldnames=["image_id","gt","pred_base","pred_lora","correct_base","correct_lora"])
        w.writeheader()
        for br, lr in zip(base_rec, lora_rec):
            w.writerow({
                "image_id": br["image_id"], "gt": br["gt"],
                "pred_base": br["pred"], "pred_lora": lr["pred"],
                "correct_base": "Y" if br["error_type"]=="none" else "N",
                "correct_lora": "Y" if lr["error_type"]=="none" else "N",
            })
    for label, recs in [("base", base_rec), ("lora", lora_rec)]:
        with (args.output_dir / ("predictions_" + label + ".csv")).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_id","gt","pred","error_type"])
            w.writeheader(); w.writerows(recs)
    _p(f"\nOutput: {args.output_dir}")
    _p("DONE!")


if __name__ == "__main__":
    main()
