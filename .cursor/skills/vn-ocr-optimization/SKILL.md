---
name: vn-ocr-optimization
description: Tối ưu hóa OCR cho VN license plates với Qwen2-VL. Dùng khi cần cải thiện accuracy, tốc độ, hoặc so sánh base vs fine-tuned.
---

# VN License Plate - OCR Optimization với Qwen2-VL

## Overview

Tối ưu hóa OCR pipeline cho VN plates với **Qwen2-VL-2B-Instruct**. So sánh base model vs fine-tuned model, optimize prompts, tune parameters.

## When to Use

- OCR accuracy thấp trên test set
- Cần so sánh base vs fine-tuned Qwen2-VL
- Muốn optimize preprocessing cho better OCR input
- Tuning generation parameters

## OCR Methods Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Qwen2-VL (base)** | Pre-trained VLM, handles variations | Cần prompt engineering | Baseline |
| **Qwen2-VL (fine-tuned)** | Tối ưu cho VN plates | Cần training data | Production |
| **Ensemble** | Robust | Complex, slower | Maximum accuracy |

## The Optimization Workflow

```
1. BASELINE ──→ 2. PREPROCESS ──→ 3. COMPARE ──→ 4. TUNE ──→ 5. DEPLOY
     │               │                │            │           │
     │               ▼                ▼            ▼           ▼
     │         Improve         Base vs      Adjust      Integrate
     │         crop quality    Fine-tuned    params      to pipeline
```

## Step 1: Establish Baseline

```bash
# Test base model
python scripts/run_inference.py \
    --input-dir data/test \
    --output-json outputs/base_predictions.json \
    --ocr-backend qwen \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda

# Evaluate
python scripts/eval_pipeline.py \
    --pred-csv outputs/base_predictions.csv \
    --gt-csv data/test_gt.csv
```

## Step 2: Preprocessing Optimization

### Crop Quality Check

```python
import cv2
import numpy as np

def analyze_crop_quality(crops_dir):
    """Check if crops are suitable for OCR"""
    issues = []
    
    for crop_path in os.listdir(crops_dir):
        crop = cv2.imread(crop_path)
        
        # Check resolution
        if crop.shape[0] < 32 or crop.shape[1] < 64:
            issues.append(f"{crop_path}: Too small")
        
        # Check aspect ratio (VN plates: ~4.5:1)
        aspect = crop.shape[1] / crop.shape[0]
        if aspect < 3 or aspect > 6:
            issues.append(f"{crop_path}: Unusual aspect ratio {aspect:.2f}")
        
        # Check blur
        laplacian = cv2.Laplacian(crop, cv2.CV_64F).var()
        if laplacian < 100:
            issues.append(f"{crop_path}: Blurry (variance={laplacian:.1f})")
    
    return issues
```

### Preprocessing for Qwen2-VL

```python
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_for_qwen(crop):
    """Preprocess crop for Qwen2-VL"""
    # Convert to PIL if numpy
    if isinstance(crop, np.ndarray):
        pil_img = Image.fromarray(crop)
    else:
        pil_img = crop
    
    # Resize to reasonable size (Qwen2-VL works well with 448-896px)
    target_size = 448
    if max(pil_img.size) > target_size:
        ratio = target_size / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
    
    # Optional: enhance contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    return pil_img
```

## Step 3: Compare Base vs Fine-tuned

### System Prompt Optimization

```python
# Prompt variations to test
PROMPTS = {
    "simple": "Đọc biển số xe trong ảnh:",
    "detailed": "Bạn là hệ thống nhận diện biển số xe Việt Nam. Đọc biển số xe trong ảnh và chỉ trả về kết quả. Định dạng: [mã tỉnh][chữ cái loại][số]. Ví dụ: 30G112345",
    "strict": "Đọc biển số xe. Chỉ trả về text, không giải thích. Nếu không chắc chắn, ghi UNREADABLE.",
}

def test_prompts(model_name, test_crops, gt_texts):
    """Compare different prompts"""
    results = {}
    
    for prompt_name, prompt in PROMPTS.items():
        predictions = []
        for crop in test_crops:
            result = qwen_ocr(crop, prompt=prompt, model=model_name)
            predictions.append(result)
        
        cer = calculate_cer(predictions, gt_texts)
        results[prompt_name] = {"cer": cer, "predictions": predictions}
    
    return results
```

### Temperature and Top-p Sampling

```python
def generate_with_params(model, processor, image, params):
    """Test different generation parameters"""
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, 
                                     {"type": "text", "text": "Đọc biển số:"}]}
    ]
    
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=params.get("max_new_tokens", 32),
            temperature=params.get("temperature", 0.1),
            top_p=params.get("top_p", 0.9),
            do_sample=params.get("do_sample", False),
        )
    
    return processor.batch_decode(
        generated_ids[:, len(inputs.input_ids[0]):], 
        skip_special_tokens=True
    )[0]

# Test configurations
CONFIGS = [
    {"temperature": 0.1, "do_sample": False},  # Greedy
    {"temperature": 0.3, "do_sample": True},
    {"temperature": 0.5, "do_sample": True},
    {"temperature": 0.7, "do_sample": True},
]
```

## Step 4: Hyperparameter Tuning

### Generation Parameters

```python
# Key parameters to tune
GEN_PARAMS = {
    "max_new_tokens": [16, 32, 64],  # VN plates are 7-9 chars
    "temperature": [0.0, 0.1, 0.3],   # Lower = more deterministic
    "top_p": [0.9, 0.95, 1.0],        # Nucleus sampling
    "repetition_penalty": [1.0, 1.1],  # Penalize repeats
}
```

### Finding Optimal Parameters

```python
from itertools import product

def grid_search_params(model, test_crops, gt_texts):
    """Find optimal generation parameters"""
    best_cer = float('inf')
    best_params = None
    
    for max_tok, temp, top_p, rep_pen in product(
        GEN_PARAMS["max_new_tokens"],
        GEN_PARAMS["temperature"],
        GEN_PARAMS["top_p"],
        GEN_PARAMS["repetition_penalty"],
    ):
        params = {
            "max_new_tokens": max_tok,
            "temperature": temp,
            "top_p": top_p,
            "repetition_penalty": rep_pen,
            "do_sample": temp > 0,
        }
        
        predictions = [generate_with_params(model, processor, crop, params) 
                      for crop in test_crops]
        cer = calculate_cer(predictions, gt_texts)
        
        if cer < best_cer:
            best_cer = cer
            best_params = params
    
    return best_params, best_cer
```

### Postprocess Rules

```python
from src.postprocess.plate_rules import (
    normalize_plate_text,
    is_valid_vn_plate,
    advanced_repair_ocr_text,
)

def postprocess_qwen_output(raw_text):
    """Postprocess Qwen2-VL output"""
    # Normalize
    normalized = normalize_plate_text(raw_text)
    
    # Fix common VLM errors
    repaired = advanced_repair_ocr_text(normalized)
    
    # Validate
    if is_valid_vn_plate(repaired):
        return repaired
    
    # Return best candidate
    return repaired
```

## Step 5: Integration

### Pipeline Integration

```python
from src.ocr import Qwen2VLPlateOcr
from src.postprocess.plate_rules import advanced_repair_ocr_text

class VNPlateOCR:
    def __init__(self, model_name="username/vn-plate-qwen2-vl-2b"):
        self.ocr = Qwen2VLPlateOcr(model_name)
    
    def recognize(self, plate_crop):
        # Get raw prediction
        result = self.ocr.recognize(plate_crop)
        
        # Postprocess
        text = advanced_repair_ocr_text(result.text_raw)
        
        return {
            "text_raw": result.text_raw,
            "text_norm": text,
            "confidence": result.ocr_score,
        }
```

### Performance Benchmarking

```python
import time

def benchmark_ocr(ocr, test_crops, num_runs=10):
    """Benchmark OCR performance"""
    times = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        for crop in test_crops:
            ocr.recognize(crop)
        elapsed = (time.perf_counter() - start) / len(test_crops) * 1000
        times.append(elapsed)
    
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }
```

## Verification

```bash
# Final evaluation
python scripts/eval_pipeline.py \
    --pred-csv outputs/optimized_ocr.csv \
    --gt-csv data/test_gt.csv

# Expected output:
# OCR Method: Qwen2-VL-2B-Instruct (fine-tuned)
# Preprocessing: resize to 448px + contrast 1.2x
# CER: 0.023 (base: 0.087)
# WER: 0.087 (base: 0.234)
# Plate Accuracy: 91.3% (base: 76.5%)
# Mean Latency: 145ms (base: 120ms)
```

## Red Flags

- Not comparing to baseline
- Changing multiple things at once
- Ignoring preprocessing quality
- No error analysis
- Overfitting to specific test images
- Not testing on held-out data
- Fine-tuning without enough training data (< 100 samples)
