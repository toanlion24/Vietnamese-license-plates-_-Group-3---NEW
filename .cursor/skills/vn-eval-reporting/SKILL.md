---
name: vn-eval-reporting
description: Đánh giá và báo cáo performance cho VN plate pipeline. Dùng khi cần generate reports, export metrics, hoặc compare experiments.
---

# VN License Plate - Evaluation & Reporting

## Overview

Standardized evaluation workflow và reporting cho VN plate recognition. Metrics, error analysis, experiment tracking.

## When to Use

- Evaluate model/pipeline performance
- Generate comparison reports
- Export error analysis
- Create thesis/presentation results

## Standard Metrics

| Metric | Formula | Target | Meaning |
|--------|---------|--------|---------|
| **CER** | Levenshtein(pred, gt) / len(gt) | < 0.05 | Character-level error |
| **WER** | Word errors / word count | < 0.10 | Word-level error |
| **Plate Accuracy** | Exact matches / total | > 90% | Full plate match |
| **Detection Rate** | Detected / total plates | > 95% | Can find plates |
| **Mean Latency** | Average ms per image | < 100ms | Speed |

## The Evaluation Workflow

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  1. PREPARE ──→ 2. RUN ──→ 3. ANALYZE ──→ 4. REPORT │
│       │            │            │            │        │
│       ▼            ▼            ▼            ▼        │
│  Manifest    Inference    Error          Markdown    │
│  + GT        + Logging     Analysis       + Charts   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Step 1: Prepare Evaluation Data

### Manifest Structure

```python
# Required manifest format
manifest = pd.DataFrame({
    'image_id': ['001', '002', ...],
    'image_path': ['data/test/001.jpg', ...],
    'plate_text': ['12A-12345', ...],      # Ground truth
    'province': ['12A', ...],               # Optional breakdown
    'serial': ['12345', ...],
    'split': ['test', ...]                  # train/val/test
})

# Save manifest
manifest.to_csv('data/manifests/test.csv', index=False)
```

### Bootstrap from Folder

```bash
# Create manifest from folder with GT file
python scripts/build_manifest_from_folder.py \
    --image-dir data/test \
    --gt-file data/gt.txt \
    --output-csv data/manifests/test.csv

# From sequential images (GT in filename)
python scripts/bootstrap_gt_csv_from_folder.py \
    --folder data/test \
    --pattern "*.jpg" \
    --gt-in-filename \
    --output-csv data/manifests/test.csv
```

## Step 2: Run Evaluation

### Standard Inference

```bash
# Run pipeline on test set
python scripts/run_infer.py \
    --manifest data/manifests/test.csv \
    --detector-backend yolov8 \
    --detector-model weights/yolov8_plate.pt \
    --ocr-backend trocr \
    --ocr-model outputs/trocr_tuned \
    --output-json outputs/predictions.json \
    --output-csv outputs/predictions.csv \
    --device cuda

# Run with timing
python scripts/run_infer.py \
    --manifest data/manifests/test.csv \
    --output-json outputs/predictions.json \
    --log-timing

# Output timing.json:
# {
#   "total_images": 100,
#   "total_time_ms": 4523,
#   "mean_latency_ms": 45.2,
#   "p50_ms": 42.1,
#   "p95_ms": 78.3,
#   "p99_ms": 123.4
# }
```

### Manifest-based Inference

```bash
# Run with manifest (includes GT for auto-evaluation)
python scripts/run_buoi4_manifest_inference.py \
    --manifest data/manifests/buoi4_test.csv \
    --detector-backend deepsolo \
    --ocr-backend trocr \
    --run-metrics \
    --metrics-json reports/buoi4_metrics.json \
    --report-md reports/buoi4_results.md
```

## Step 3: Analyze Errors

### Character Error Analysis

```bash
# Export errors by region (tỉnh/chữ/serial)
python scripts/export_char_errors_csv.py \
    --pred-csv outputs/predictions.csv \
    --gt-csv data/gt.csv \
    --output-csv reports/char_errors.csv

# The output includes:
# image_id, gt, pred, char_errors, region, position
# Regions: province (0-1), letter (3), serial (5-9)
```

### Error Pattern Analysis

```python
# Analyze error patterns
def analyze_errors(pred_csv, gt_csv):
    df = pd.merge(pred_csv, gt_csv, on='image_id', suffixes=('_pred', '_gt'))
    
    # Overall metrics
    df['exact_match'] = df['pred'] == df['gt']
    df['cer'] = df.apply(lambda r: levenshtein(r['pred'], r['gt']) / len(r['gt']), axis=1)
    
    # By region
    for region in ['province', 'letter', 'serial']:
        # Count errors by region
        pass
    
    return {
        'accuracy': df['exact_match'].mean(),
        'cer_mean': df['cer'].mean(),
        'error_patterns': error_patterns
    }
```

### Hard Cases Export

```bash
# Export hardest cases for analysis
python scripts/export_buoi4_hard_cases.py \
    --pred-csv outputs/predictions.csv \
    --gt-csv data/gt.csv \
    --output-dir reports/hard_cases \
    --top-n 20

# Creates:
# reports/hard_cases/
#   ├── case_001_cer0.8.jpg  (visual)
#   ├── case_002_cer0.6.jpg
#   └── hard_cases_manifest.csv
```

### Detection Analysis

```bash
# Analyze detection patterns
python scripts/analyze_detection_pattern.py \
    --manifest data/manifests/test.csv \
    --pred-json outputs/detections.json \
    --output-csv reports/detection_analysis.csv

# Check for:
# - Missed detections
# - False positives
# - Small plate misses
# - Rotation sensitivity
```

## Step 4: Generate Reports

### A/B Experiment Comparison

```bash
# Compare two configurations
python scripts/run_buoi4_experiments.py \
    --config-a-csv outputs/baseline.csv \
    --config-a-name "EasyOCR" \
    --config-b-csv outputs/optimized.csv \
    --config-b-name "TrOCR" \
    --metrics-json reports/ab_metrics.json \
    --report-md reports/ab_comparison.md

# Report format:
# ## A/B Comparison Results
# 
# | Metric      | EasyOCR | TrOCR  | Δ     |
# |-------------|---------|--------|-------|
# | CER         | 0.045   | 0.023  | -49%  |
# | WER         | 0.156   | 0.087  | -44%  |
# | Accuracy    | 84.4%   | 91.3%  | +6.9% |
```

### Visual Report

```bash
# Generate visual report
python scripts/create_buoi4_visual_report.py \
    --pred-csv outputs/predictions.csv \
    --gt-csv data/gt.csv \
    --output-dir reports/visual

# Creates:
# reports/visual/
#   ├── sample_predictions.html  (interactive)
#   ├── error_distribution.png
#   ├── confusion_matrix.png
#   └── metrics_summary.json
```

### Video Comparison

```bash
# Compare two videos
python scripts/compare_video_ocr_json.py \
    --video-a videos/test1.mp4 \
    --video-b videos/test2.mp4 \
    --output-json reports/video_compare.json

# Summarize
python scripts/summarize_video_compare_csv.py \
    --compare-json reports/video_compare.json \
    --output-md reports/video_summary.md
```

## Report Template

```markdown
# Báo Cáo Đánh Giá - Nhận Diện Biển Số VN

## 1. Tổng Quan

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| CER | 0.023 | < 0.05 | ✅ Pass |
| WER | 0.087 | < 0.10 | ✅ Pass |
| Plate Accuracy | 91.3% | > 90% | ✅ Pass |
| Mean Latency | 45ms | < 100ms | ✅ Pass |

## 2. Chi Tiết Theo Region

| Region | Accuracy | Top Errors |
|--------|----------|-----------|
| Tỉnh (2 chars) | 95.2% | 0↔O, 1↔I |
| Chữ (1 char) | 98.1% | Font-specific |
| Serial (5 chars) | 92.3% | 5↔S, 0↔O |

## 3. Error Analysis

### Hardest Cases
1. `IMG_001.jpg`: CER=0.8 - Low resolution, tilted
2. `IMG_002.jpg`: CER=0.6 - Partial occlusion

### Common Patterns
- Rotation > 15° → significant accuracy drop
- Resolution < 100px height → detector miss

## 4. Comparison with Previous Run

| Metric | Baseline | Current | Improvement |
|--------|----------|---------|-------------|
| CER | 0.045 | 0.023 | -49% |
| Accuracy | 84.4% | 91.3% | +6.9% |

## 5. Recommendations

1. Collect more data for tilted plates
2. Fine-tune on low-resolution samples
3. Add data augmentation for rotation
```

## Metrics Calculation

```python
import Levenshtein

def calculate_metrics(predictions, ground_truth):
    """Calculate standard VN plate metrics"""
    results = []
    
    for pred, gt in zip(predictions, ground_truth):
        cer = Levenshtein.distance(pred, gt) / len(gt)
        exact = pred == gt
        results.append({
            'pred': pred,
            'gt': gt,
            'cer': cer,
            'exact_match': exact
        })
    
    df = pd.DataFrame(results)
    
    return {
        'cer': df['cer'].mean(),
        'wer': 1 - df['exact_match'].mean(),  # WER ≈ 1 - accuracy
        'accuracy': df['exact_match'].mean(),
        'total_samples': len(df),
        'correct': df['exact_match'].sum()
    }
```

## Red Flags

- Reporting on training data
- Not splitting train/val/test
- Ignoring edge cases
- Cherry-picking best examples
- No statistical significance
- Not documenting config

## Verification

After evaluation:

- [ ] Evaluated on held-out test set
- [ ] Metrics calculated correctly
- [ ] Error analysis done
- [ ] Hard cases documented
- [ ] Report generated
- [ ] Config documented
