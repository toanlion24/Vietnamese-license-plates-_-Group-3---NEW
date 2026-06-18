---
name: vn-context-engineering
description: Tối ưu hóa context cho AI assistant khi làm việc với dự án VN plate recognition. Dùng khi bắt đầu session mới, chuyển task, hoặc cần cải thiện output quality.
---

# VN License Plate - Context Engineering

## Overview

Cung cấp đúng context cho AI assistant để maintain quality và consistency trong dự án VN plate recognition.

## The Context Hierarchy

```
┌─────────────────────────────────────┐
│  1. Rules (SKILL.md, .cursor/rules) │ ← Always loaded
├─────────────────────────────────────┤
│  2. Spec / Project docs              │ ← Loaded per session
├─────────────────────────────────────┤
│  3. Relevant Source Files            │ ← Loaded per task
├─────────────────────────────────────┤
│  4. Error Output / Metrics          │ ← Loaded per iteration
├─────────────────────────────────────┤
│  5. Conversation History             │ ← Accumulates
└─────────────────────────────────────┘
```

## Level 1: Rules Files

### Available Skills

| Skill | When to Use |
|-------|-------------|
| `vn-incremental-implementation` | Building new features |
| `vn-debugging-cv` | Debugging OCR/detector issues |
| `vn-code-review` | Before committing |
| `vn-model-training` | Training/fine-tuning models |
| `vn-ocr-optimization` | Improving OCR accuracy |
| `vn-eval-reporting` | Generating reports |
| `vn-git-workflow` | Version control |
| `vn-planning` | Planning new phases |

### Project Conventions

**Directory Structure:**
```
ComputerVisionNew/
├── src/
│   ├── io/           # Frame readers
│   ├── detector/     # YOLO, DeepSolo wrappers
│   ├── preprocess/   # Crop, rectify, denoise
│   ├── ocr/         # EasyOCR, TrOCR adapters
│   ├── postprocess/  # Regex, normalization
│   ├── pipeline/     # End-to-end orchestration
│   ├── eval/         # Metrics, error analysis
│   └── app/          # CLI, webcam, demo
├── scripts/          # Training, inference, evaluation
├── data/            # Manifests, GT, outputs
├── configs/         # Model configs
├── weights/         # Trained models
├── reports/         # Evaluation reports
└── docs/            # Notebooks, thesis docs
```

**Code Conventions:**
- Python 3.10+, snake_case
- Type hints for public APIs
- Dataclasses for stage payloads
- Pure functions for pre/post-processing
- Log per-stage latency

## Level 2: Spec Documents

### Key Documents

| Document | Contains |
|----------|----------|
| `README.md` | Project overview, quick start |
| `docs/KIEM_TRA_DE_TAI_VA_HOI_DONG.md` | Đề tài checklist |
| `docs/pre-commit-checklist.md` | Pre-commit steps |
| `data/manifests/README.md` | Data format |
| `.cursor/plans/*.md` | Active plans |

### Before Starting Task

Load relevant spec section:
```
TASK: Fine-tune TrOCR
RELEVANT DOCS:
- scripts/train_trocr.py (existing training script)
- configs/trocr/README.md (config format)
- src/ocr/trocr_adapter.py (current implementation)
```

## Level 3: Relevant Source Files

### Task-Specific Context

```markdown
TASK: Improve OCR accuracy

RELEVANT FILES:
1. src/ocr/base.py         - OCR interface
2. src/ocr/trocr_adapter.py - TrOCR implementation
3. src/ocr/easyocr_adapter.py - EasyOCR for comparison

PATTERN TO FOLLOW:
- See src/ocr/trocr_adapter.py:30-50 for current inference pattern
- Use dataclass OCRResult for output

CONSTRAINTS:
- Must maintain fallback chain (TrOCR -> EasyOCR)
- Must log confidence scores
```

## Level 4: Error Output

When debugging, provide specific errors:

```
GOOD:
OCR error on IMG_004.jpg:
- Expected: "12A-12345"
- Got: "12A-123S5"
- Error: Character 7 (position 6 in serial) confused '5' -> 'S'
- Confidence: 0.72 (low)

BAD:
"OCR doesn't work"
```

## Level 5: Session Management

### Start of Session

```markdown
SESSION START: VN Plate Pipeline - Buổi 4 Experiments

OBJECTIVE: Compare TrOCR vs EasyOCR, report metrics

PREVIOUS PROGRESS:
- YOLOv8 detector trained (mAP=0.94)
- TrOCR fine-tuned on 500 synthetic plates
- Baseline EasyOCR: CER=0.045, Accuracy=84%

FILES TO WORK WITH:
- scripts/run_buoi4_manifest_inference.py
- scripts/export_char_errors_csv.py
- scripts/run_buoi4_experiments.py

NEXT:
1. Run inference with TrOCR on test set
2. Compare with EasyOCR baseline
3. Export error analysis
4. Update Buổi 4 report
```

### Mid-Session Summary

```
PROGRESS UPDATE:

✅ TrOCR inference complete
✅ Error analysis exported
✅ Comparison report generated

CURRENT STATUS:
- TrOCR: CER=0.023, Accuracy=91.3% (significant improvement!)
- EasyOCR: CER=0.045, Accuracy=84%

OBSERVATIONS:
- TrOCR handles tilted plates better
- Serial confusion (5↔S) still common

NEXT:
- Update report with new results
- Consider ensemble approach for remaining errors
```

## Confusion Management

### When Spec Conflicts with Code

```
SPEC says:     Use EasyOCR as primary OCR
CODE has:      TrOCR as primary (better accuracy)

OPTIONS:
A) Follow spec → Revert to EasyOCR (accuracy drops)
B) Follow code → Update spec with justification
C) Ask → Get clarification

→ Recommend B: TrOCR significantly better, document rationale
```

### When Requirements Unclear

```
MISSING INFO:
Spec defines "plate accuracy" but doesn't specify exact formula.

Options:
A) Exact match only (pred == gt)
B) Allow format normalization (12A12345 == 12A-12345)
C) Allow CER threshold (CER < 0.1 = pass)

→ Recommend B for production, A for strict evaluation
```

## Anti-Patterns

| Anti-Pattern | Problem | Fix |
|---|---|---|
| Context starvation | Agent invents APIs | Load skill + source files |
| Context flooding | Agent loses focus (>5000 lines) | Include only relevant files |
| Stale context | Agent uses old patterns | Start fresh or summarize |
| Implicit knowledge | Agent doesn't know conventions | Add to rules/skill |

## Verification

After setting up context:

- [ ] Relevant skill(s) loaded
- [ ] Source files read before editing
- [ ] Metrics baseline available
- [ ] Task objective clear
- [ ] Success criteria defined
