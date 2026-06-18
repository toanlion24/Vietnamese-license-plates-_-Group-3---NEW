---
name: vn-debugging-cv
description: Debug pipeline nhận diện biển số VN. Dùng khi OCR cho kết quả sai, detector miss plates, hoặc metrics giảm sau thay đổi.
---

# VN License Plate - Debugging & Error Recovery

## Overview

Systematic debugging cho CV pipeline. Khi OCR sai, detector miss, hoặc metrics giảm — stop, triage, fix có phương pháp.

## When to Use

- OCR cho kết quả sai (wrong characters, wrong format)
- Detector miss plates hoặc false positives
- Metrics giảm sau thay đổi
- Pipeline chạy chậm bất thường
- Bất kỳ unexpected error nào

## The Stop-the-Line Rule

```
1. STOP — Không continue với new features
2. PRESERVE — Lưu error output, bad predictions
3. TRIAGE — Systematic check theo checklist
4. FIX — Sửa root cause
5. GUARD — Thêm test/regression check
6. VERIFY — Chạy lại evaluation
```

## Triage Checklist

### Step 1: Reproduce

Tái hiện lỗi một cách đáng tin cậy:

```bash
# Run on specific failing images
python scripts/run_infer.py --input-dir data/test_failing --output-json debug/pred.json

# Run with verbose/debug output
python scripts/run_infer.py --input-dir data/test_failing --debug

# Check specific image
python -c "from src.pipeline import infer; print(infer('data/test/001.jpg'))"
```

### Step 2: Localize

Xác định stage nào gây ra lỗi:

```
Lỗi ở đâu?
├── Detector miss?
│   └── Check: confidence threshold, NMS, image quality
├── Preprocessor tạo crop tệ?
│   └── Check: crop quality, aspect ratio, resolution
├── OCR sai?
│   └── Check: EasyOCR vs TrOCR, preprocessing input
├── Postprocess format sai?
│   └── Check: regex patterns, Vietnamese normalization
└── Evaluation script bug?
    └── Check: GT labels, comparison logic
```

### Step 3: Analyze Specific Error Types

#### OCR Errors by Region

```bash
# Export errors by region (tỉnh/chữ/serial)
python scripts/export_char_errors_csv.py \
    --pred-csv outputs/predictions.csv \
    --output-csv reports/char_errors.csv
```

Check patterns:
- **Tỉnh** (2 chars): Thường confuse giữa các tỉnh gần nhau
- **Chữ** (1 char): Font-specific errors
- **Serial** (5 chars): Number confusions (0/O, 1/I)

#### Common OCR Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "0" → "O" | Font style | Add to confusion set |
| "1" → "I" or "l" | Font style | Add to confusion set |
| Missing accent | Low resolution | Improve preprocessing |
| Extra chars | Background noise | Better thresholding |

### Step 4: Fix Root Cause

Fix nguyên nhân gốc, không phải symptom:

```
Symptom: OCR confuse "5" → "S" in serial

Symptom fix (BAD):
  → Hard-code replacement "S" → "5" everywhere

Root cause fix (GOOD):
  → Improve postprocess to use plate format validation
  → VN plates: serial is 5 digits, so any "S" in serial = likely "5"
```

### Step 5: Guard Against Recurrence

Thêm test case cho error đã fix:

```python
# Test postprocess for common confusions
def test_vietnamese_plate_normalization():
    # 0/O confusion
    assert normalize("12A-000.5S") == "12A-00055"
    # I/1 confusion
    assert normalize("51A-000I2") == "51A-00012"
    # Serial validation
    assert is_valid_serial("12345") == True
    assert is_valid_serial("1234S") == False  # Letter in serial position
```

### Step 6: Verify

```bash
# Run full evaluation
python scripts/eval_pipeline.py --pred-csv outputs/predictions.csv

# Compare with baseline
python scripts/run_buoi4_experiments.py \
    --config-a-csv outputs/baseline.csv \
    --config-b-csv outputs/fixed.csv \
    --report-md reports/debug_fix_report.md
```

## Safe Fallback Patterns

```python
# Fallback chain for OCR
def recognize_with_fallback(crop):
    # Try TrOCR first (usually better)
    try:
        result = trocr_recognize(crop)
        if is_confident(result):
            return result
    except:
        pass
    
    # Fallback to EasyOCR
    try:
        return easyocr_recognize(crop)
    except:
        return "ERROR"

# Conservative confidence threshold
def is_confident(text, min_confidence=0.7):
    # VN plates are 8 chars: 2 province + 1 letter + 5 serial
    if len(text) != 8:
        return False
    if text[3] not in LETTERS:  # Position 3 is letter
        return False
    return average_confidence(text) >= min_confidence
```

## Debug Commands Reference

```bash
# Debug detector output
python scripts/debug_detection.py --image data/test/001.jpg --visualize

# Debug OCR on specific crop
python scripts/debug_easyocr.py --crop crops/001_crop.jpg

# Compare OCR methods
python scripts/test_easyocr_vs_trocr.py --image data/test/001.jpg

# Analyze detection patterns
python scripts/analyze_detection_pattern.py --manifest data/manifests/test.csv

# Export error analysis
python scripts/export_char_errors_csv.py \
    --pred-csv outputs/pred.csv \
    --gt-csv data/gt.csv \
    --output-csv reports/errors.csv
```

## Red Flags

- Ignoring failing test cases
- "Fixing" by removing failing images from test set
- Multiple unrelated changes while debugging
- No regression test after fix
- Changing thresholds without measurement
- Hard-coding specific image results

## Verification

After fixing:

- [ ] Root cause identified and documented
- [ ] Fix addresses root cause
- [ ] Error cases now pass
- [ ] No regression on other cases
- [ ] Evaluation metrics improved or maintained
- [ ] Regression test added
