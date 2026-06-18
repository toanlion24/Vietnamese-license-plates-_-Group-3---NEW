---
name: vn-planning
description: Lập kế hoạch cho dự án VN plate recognition. Dùng khi có spec/r requirements và cần break thành tasks.
---

# VN License Plate - Planning & Task Breakdown

## Overview

Decompose work thành small, verifiable tasks với explicit acceptance criteria và metrics.

## When to Use

- Starting new phase (Buổi 2, 3, 4...)
- Planning thesis chapters
- Breaking large features into implementable units
- Estimating scope for progress reports

## The Planning Process

### Step 1: Understand Requirements

```markdown
# Requirement Analysis

## From Đề tài / Yêu cầu
- [ ] Mục tiêu chính
- [ ] Các mốc thời gian
- [ ] Criteria đánh giá

## From Thesis Outline
- [ ] Chapter structure
- [ ] Required experiments
- [ ] Expected results
```

### Step 2: Map Dependencies

```
Data Collection
    │
    ├── Annotations
    │       │
    │       └── Manifest Building
    │
    └── Preprocessing
            │
            ├── Detector Training
            │       │
            │       └── OCR Training
            │               │
            │               └── Pipeline Integration
            │
            └── Evaluation
                    │
                    └── Report Generation
```

### Step 3: Break into Vertical Slices

**Bad (horizontal slicing):**
```
Task 1: Collect all data
Task 2: Train all models
Task 3: Write all chapters
```

**Good (vertical slicing):**
```
Task 1: Data pipeline (collect -> annotate -> manifest)
Task 2: Baseline detector (YOLOv8 baseline)
Task 3: Baseline OCR (EasyOCR baseline)
Task 4: Evaluation baseline metrics
Task 5: Improved detector (fine-tuned)
Task 6: Improved OCR (TrOCR fine-tuned)
Task 7: Full pipeline
Task 8: Error analysis
Task 9: Chapter 3 draft
```

## Task Template

```markdown
## Task [N]: [Short Title]

**Description:** One paragraph explaining what this accomplishes.

**Acceptance Criteria:**
- [ ] [Specific, measurable condition]
- [ ] [Specific, measurable condition]

**Metrics:**
- [ ] CER < 0.05
- [ ] Accuracy > 90%
- [ ] Mean latency < 100ms

**Verification:**
```bash
python scripts/eval_pipeline.py --pred-csv outputs/predictions.csv
```

**Dependencies:** Task N-1 (or "None")

**Files Involved:**
- `src/pipeline/infer.py`
- `src/ocr/trocr_adapter.py`
- `scripts/run_infer.py`

**Scope:** [XS/S/M/L/XL]

**Estimated Time:** [X hours]
```

## Task Sizing

| Size | Scope | Example |
|------|-------|---------|
| **XS** | 1 file, 1 function | Add validation function |
| **S** | 1-2 files | New OCR adapter |
| **M** | 3-5 files, 1 stage | Implement detector |
| **L** | 5-8 files, 2+ stages | Full OCR pipeline |
| **XL** | **Too large** → Split! | Full pipeline + evaluation |

## Phased Planning

### Phase 1: Data Preparation

```markdown
## Phase 1: Data & Baseline (Buổi 1-2)

### Task 1.1: Data Collection
- [ ] Collect 100+ real VN plate images
- [ ] Annotate bounding boxes
- [ ] Create manifest CSV

### Task 1.2: Baseline Pipeline
- [ ] YOLOv8 detector (pretrained)
- [ ] EasyOCR backend
- [ ] Basic postprocessing
- [ ] Run on test set → Baseline metrics

### Checkpoint: Data & Baseline
- [ ] 100+ images annotated
- [ ] Baseline CER < 0.10
- [ ] Can run end-to-end inference
```

### Phase 2: Model Improvement

```markdown
## Phase 2: Model Training (Buổi 3-4)

### Task 2.1: Detector Training
- [ ] Fine-tune YOLOv8 on VN plates
- [ ] Evaluate mAP > 0.90
- [ ] Export weights

### Task 2.2: OCR Training
- [ ] Fine-tune TrOCR on VN plates
- [ ] Compare with EasyOCR
- [ ] Select best method

### Task 2.3: Pipeline Integration
- [ ] Integrate improved models
- [ ] Optimize preprocessing
- [ ] Add confidence thresholds

### Checkpoint: Improved Models
- [ ] CER < 0.05
- [ ] Accuracy > 90%
- [ ] Error analysis complete
```

### Phase 3: Analysis & Documentation

```markdown
## Phase 3: Analysis & Thesis (Buổi 5+)

### Task 3.1: Error Analysis
- [ ] Character-level analysis
- [ ] Regional breakdown
- [ ] Hard case documentation

### Task 3.2: Thesis Writing
- [ ] Chapter 1: Introduction
- [ ] Chapter 2: Related Work
- [ ] Chapter 3: Methodology
- [ ] Chapter 4: Experiments
- [ ] Chapter 5: Conclusion

### Task 3.3: Final Defense Prep
- [ ] Demo ready
- [ ] Q&A prepared
- [ ] Slides complete
```

## Progress Tracking

```markdown
# Progress: VN License Plate Recognition

## Overall: ████████░░ 80%

## By Phase:
- [██████░░░░] Phase 1: Data & Baseline (60%)
- [████████░░] Phase 2: Model Training (80%)
- [██░░░░░░░░] Phase 3: Analysis & Thesis (20%)

## Current Task: Task 2.2 - OCR Training
- Started: 2024-03-15
- Expected: 2024-03-20
- Status: In progress

## Blockers:
- None

## Next:
- Task 2.3: Pipeline Integration
```

## Red Flags

- Tasks without acceptance criteria
- No metrics defined
- Tasks larger than "L"
- No checkpoint between phases
- Dependencies not considered
- No time estimation
