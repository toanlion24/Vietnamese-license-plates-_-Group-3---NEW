---
name: vn-code-review
description: Review code cho ML pipeline nhận diện biển số. Dùng trước khi merge bất kỳ thay đổi nào.
---

# VN License Plate - Code Review

## Overview

Multi-axis review cho ML code. Check correctness, reproducibility, performance, và reproducibility trước khi merge.

## The Five-Axis Review

### 1. Correctness

Code có làm đúng như spec/kwargs?

- Inference logic đúng flow?
- Preprocessing/Postprocessing đúng format?
- Metrics calculation đúng công thức?
- Edge cases handled (empty image, no detection, etc.)?

```python
# Check CER calculation
def calculate_cer(pred, gt):
    # Levenshtein distance / len(gt)
    # Should be 0.0 for exact match
    # Should be 1.0 for completely wrong
    
# BAD: Wrong normalization
    return levenshtein(pred, gt) / len(pred)  # Wrong!

# GOOD: Correct normalization
    return levenshtein(pred, gt) / len(gt)  # Correct!
```

### 2. Reproducibility

Code có thể reproduce được kết quả?

- Random seeds set đúng?
- Model weights loaded correctly?
- Config không hard-coded?
- Dependencies documented?

```python
# GOOD: Reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# BAD: Non-reproducible
model = YOLO("yolov8.pt")  # No seed
```

### 3. Performance

Code có performance issues?

- Batch size reasonable?
- Memory usage bounded?
- Latency logged?
- No N+1 in loops?

```python
# BAD: Process one by one
for img_path in images:
    result = ocr.recognize(img_path)  # Slow

# GOOD: Batch if supported
results = ocr.recognize_batch(images)  # Faster
```

### 4. ML-Specific Checks

- Model config matches training config?
- Inference mode correct (eval vs train)?
- Device handling (CPU/GPU)?
- Input preprocessing matches training?

```python
# GOOD: Proper inference mode
model.eval()
with torch.no_grad():
    output = model(input)

# BAD: Forgot eval mode
output = model(input)  # Running in train mode!
```

### 5. Code Quality

- Clear function names?
- Type hints present?
- Docstrings for complex logic?
- Error messages helpful?

## Change Sizing

```
~100 lines changed   → Good. Reviewable.
~300 lines changed   → Acceptable if single logical change.
~1000+ lines        → Too large. Split.
```

## Review Checklist

```markdown
## Review: [PR Title]

### Context
- [ ] Hiểu change làm gì và tại sao

### Correctness
- [ ] Logic đúng với spec
- [ ] Edge cases handled
- [ ] Metrics calculation correct

### Reproducibility
- [ ] Seeds set
- [ ] Configs documented
- [ ] Dependencies listed

### Performance
- [ ] No obvious bottlenecks
- [ ] Latency acceptable
- [ ] Memory bounded

### ML-Specific
- [ ] eval() mode for inference
- [ ] Preprocessing matches training
- [ ] Device handling correct

### Quality
- [ ] Clear naming
- [ ] Type hints
- [ ] Docstrings for complex logic

### Verification
- [ ] Evaluation script chạy được
- [ ] Metrics measured
- [ ] No regression

### Verdict
- [ ] **Approve** — Ready to merge
- [ ] **Request changes** — Issues must be addressed
```

## Dead Code Hygiene

Sau khi refactor, check cho dead code:

```
DEAD CODE FOUND:
- src/ocr/old_easyocr.py — replaced by new adapter
- scripts/train_deprecated.py — not used anymore
→ Safe to remove these?
```

## Severity Labels

| Prefix | Meaning |
|--------|---------|
| *(no prefix)* | Required — must fix |
| **Nit:** | Optional — formatting, style |
| **Consider:** | Worth considering |
| **FYI:** | Context only |

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "Code works, metrics OK" | Nhưng có thể không reproducible |
| "I'll set seeds later" | Sẽ không set |
| "It runs fine on my machine" | GPU vs CPU khác nhau |

## Red Flags

- No evaluation before/after
- Hard-coded paths instead of relative
- Missing seed setting
- eval() mode not set for inference
- Mixing train/inference logic
- No error handling for edge cases
