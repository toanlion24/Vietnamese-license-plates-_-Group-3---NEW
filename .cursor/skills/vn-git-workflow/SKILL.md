---
name: vn-git-workflow
description: Git workflow cho dự án VN plate recognition. Dùng khi commit, tạo branch, hoặc cần organize work across sessions.
---

# VN License Plate - Git Workflow

## Overview

Structured git workflow với focus trên ML reproducibility, experiment tracking, và thesis documentation.

## When to Use

- Commit any code change
- Create experiment branches
- Organize work across sessions
- Document thesis progress

## Branch Naming

```
main ──●──●──●──●──●──●──●──●──●──  (always stable)

feature/
  ├── feature/yolo-detector-v2
  ├── feature/trocr-finetune
  └── feature/preprocessing-improvements

experiment/
  ├── experiment/compare-ocr-methods
  └── experiment/low-res-detection

thesis/
  ├── thesis/buoi-4-experiments
  └── thesis/performance-analysis
```

## Commit Structure

### Format

```
<type>: <short description>

<body explaining why and what>
```

### Types

| Type | When to Use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructure without behavior change |
| `train` | Model training results (include metrics in body) |
| `eval` | Evaluation results (include metrics in body) |
| `exp` | Experiment results |
| `docs` | Documentation |
| `scripts` | Utility scripts |
| `data` | Data-related changes |

### Examples

```bash
# Good commit for feature
git commit -m "feat: add TrOCR as OCR backend option

- Add TrOCRProcessor and VisionEncoderDecoderModel
- Implement fallback chain: TrOCR -> EasyOCR
- Add --ocr-backend flag to run_infer.py

CER: 0.045 -> 0.023 (-49%)
Accuracy: 84% -> 91%"

# Good commit for experiment
git commit -m "exp: compare EasyOCR vs TrOCR on 100 test images

Results:
- EasyOCR: CER=0.045, Accuracy=84%
- TrOCR: CER=0.023, Accuracy=91%

Conclusion: TrOCR significantly better for VN plates"

# Good commit for thesis
git commit -m "docs: update Buổi 4 experiment results

- Add comparison tables
- Update figures
- Fix methodology section"
```

## Experiment Tracking

### Structure

```
experiments/
├── 2024-03-15_easyocr_baseline/
│   ├── config.json
│   ├── predictions.csv
│   ├── metrics.json
│   └── notes.md
├── 2024-03-16_trocr_finetune/
│   ├── config.json
│   ├── predictions.csv
│   ├── metrics.json
│   └── notes.md
└── README.md  # Index of all experiments
```

### Experiment README Template

```markdown
# Experiment: [Title]

## Date: YYYY-MM-DD

## Objective
[What we're trying to learn/improve]

## Configuration
- Detector: YOLOv8 (weights/yolov8_plate.pt)
- OCR: TrOCR fine-tuned (outputs/trocr_tuned)
- Dataset: data/test (100 images)

## Results

| Metric | Value |
|--------|-------|
| CER | 0.023 |
| WER | 0.087 |
| Accuracy | 91.3% |
| Mean Latency | 45ms |

## Observations
- TrOCR handles rotated plates better
- EasyOCR faster but less accurate

## Next Steps
- [ ] Fine-tune on more data
- [ ] Add more augmentations
```

## The Save Point Pattern

```
Start work
    │
    ├── Make small change
    │   ├── Run evaluation
    │   ├── Metrics improved/maintained? → Commit → Continue
    │   └── Metrics worse? → Revert → Investigate
    │
    └── Repeat until feature complete
```

## Pre-Commit Checklist

```bash
# 1. Check what you're about to commit
git diff --staged

# 2. Ensure no secrets
git diff --staged | Select-String -Pattern "password|secret|api_key|token"

# 3. Run tests/evaluation
python scripts/eval_pipeline.py --pred-csv outputs/predictions.csv

# 4. Check notebook outputs stripped
python scripts/strip_ipynb_outputs.py docs/*.ipynb

# 5. Update experiment log if needed
```

## Notebook Workflow

```bash
# Before commit: strip outputs
python scripts/strip_ipynb_outputs.py docs/buoi-4-analysis.ipynb

# Commit notebooks as documentation
git commit -m "docs: update Buổi 4 analysis notebook

- Added error analysis section
- Updated figures
- Stripped outputs for clean history"
```

## Data Handling

### DO Commit
- Manifests (CSV/JSON)
- Scripts
- Configs
- Experiment logs
- Notebooks (output stripped)

### DON'T Commit
- Model weights (large files) → use git lfs or external storage
- Raw images (large files) → reference in manifest
- Virtual environments
- `.env` files
- Cache directories (`__pycache__/`, `.cache/`)

### .gitignore Template

```
# Data
data/raw/
data/synthetic/
weights/
models/
outputs/
experiments/*/

# Python
__pycache__/
*.pyc
venv/
env/

# Notebooks
*.ipynb_checkpoints/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

## Red Flags

- Committing model weights to git (use LFS or external)
- No evaluation before/after commits
- Committing notebooks with outputs
- Large uncommitted changes
- Mixing thesis writing with code changes
- No experiment documentation
