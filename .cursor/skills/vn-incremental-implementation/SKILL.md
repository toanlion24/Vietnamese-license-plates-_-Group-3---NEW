---
name: vn-incremental-implementation
description: Triển khai pipeline nhận diện biển số VN theo từng bước nhỏ. Dùng khi implement detector/OCR/pipeline mới, khi thay đổi nhiều files, hoặc khi cần build feature lớn theo từng giai đoạn.
---

# VN License Plate - Incremental Implementation

## Overview

Build the VN license plate pipeline in thin vertical slices — implement one stage (detector/OCR/postprocess), test it, verify it, then expand. Each increment should leave the system in a working, testable state with measurable metrics.

## When to Use

- Implement any multi-file change (detector, OCR, pipeline)
- Building a new feature from task breakdown
- Fine-tuning models (TrOCR, YOLO)
- Refactoring existing code

## The Increment Cycle

```
┌──────────────────────────────────────┐
│  Implement ──→ Test ──→ Verify ──┐   │
│      ▲                            │   │
│      └───── Commit ◄──────────────┘   │
│              │                       │
│              ▼                       │
│          Next slice                  │
└──────────────────────────────────────┘
```

For each slice:

1. **Implement** the smallest complete piece (e.g., one OCR backend)
2. **Test** — run evaluation on validation set
3. **Verify** — confirm metrics (CER/WER/accuracy) improved or maintained
4. **Commit** — save with descriptive message
5. **Next slice** — continue

## Pipeline Stages (Vertical Slices)

```
Stage 1: Detector (YOLO/DeepSolo)
    → Can detect plates in images

Stage 2: Preprocessor (crop, rectify, denoise)
    → Clean crops for OCR

Stage 3: OCR Engine (EasyOCR/TrOCR)
    → Raw text output

Stage 4: Postprocessor (Vietnamese normalization, regex)
    → Corrected plate format

Stage 5: Evaluator (CER/WER/accuracy)
    → Measurable metrics

Stage 6: CLI/App (image/webcam/video)
    → End-to-end demo
```

## Implementation Rules

### Rule 0: Measure First

Before changing anything, establish a baseline:

```bash
python scripts/eval_pipeline.py --pred-csv outputs/baseline.csv
# Record: CER=X, WER=Y, Accuracy=Z
```

### Rule 1: One Stage at a Time

Each increment changes one logical stage:

```
✅ GOOD: "Add TrOCR as OCR backend option"
❌ BAD: "Add TrOCR + improve preprocessing + fix postprocess regex"
```

### Rule 2: Keep Metrics Visible

After each increment, run evaluation and record:

```
Slice N: [Description]
- CER: 0.XX → 0.XX (Δ=±0.XX)
- WER: 0.XX → 0.XX (Δ=±0.XX)  
- Plate Accuracy: XX% → XX% (Δ=±X%)
- Runtime: XXXms/img
```

### Rule 3: Fallback Strategy

Always have a fallback for each stage:

```python
# OCR fallback chain
def recognize_plate(crop):
    try:
        return trocr_recognize(crop)  # Primary
    except Exception:
        return easyocr_recognize(crop)  # Fallback
```

### Rule 4: Log Per-Stage Latency

```python
def pipeline_infer(image_path):
    t0 = time.time()
    boxes = detector.detect(image_path)
    t1 = time.time()
    
    crops = [preprocess(image, box) for box in boxes]
    t2 = time.time()
    
    texts = [ocr.recognize(crop) for crop in crops]
    t3 = time.time()
    
    plates = [postprocess(text) for text in texts]
    
    logger.info(f"detector={t1-t0:.0f}ms, preprocess={t2-t1:.0f}ms, ocr={t3-t2:.0f}ms")
    return plates
```

## Verification

After each increment:

- [ ] Metrics measured and recorded
- [ ] No regression in existing stages
- [ ] Pipeline runs end-to-end
- [ ] Code follows project conventions (see `src/` structure)
- [ ] Committed with descriptive message

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "I'll test everything at the end" | Metrics compound. A regression in Slice 1 makes Slices 2-5 appear worse. |
| "This OCR change is obvious" | Run evaluation. TrOCR on VN plates may behave unexpectedly. |
| "I'll optimize later" | Add timing logs now. Latency matters for video inference. |

## Red Flags

- Changing multiple pipeline stages in one commit
- No metrics before/after comparison
- Hard-coded paths instead of using project root resolution
- Mixing new code with refactoring
- Committing without running evaluation
