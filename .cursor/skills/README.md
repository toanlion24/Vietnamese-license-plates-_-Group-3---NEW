# VN License Plate Recognition - Skills Index

Các skills cho AI assistant làm việc với dự án nhận diện biển số VN.

## Skills Overview

| Skill | Description | When to Use |
|-------|-------------|-------------|
| `vn-context-engineering` | Setup context cho AI sessions | Bắt đầu session, chuyển task |
| `vn-incremental-implementation` | Build pipeline theo từng bước nhỏ | Implement features mới |
| `vn-debugging-cv` | Debug OCR/detector errors | Khi có lỗi hoặc metrics giảm |
| `vn-code-review` | Review code trước merge | Trước khi commit |
| `vn-model-training` | Training workflow | Fine-tune models |
| `vn-ocr-optimization` | Tối ưu OCR accuracy | Cải thiện OCR |
| `vn-eval-reporting` | Evaluation và reporting | Generate reports, metrics |
| `vn-git-workflow` | Git conventions | Commit, branch management |
| `vn-planning` | Lập kế hoạch | Planning phases, tasks |
| `vn-source-driven` | Documentation verification | Verify against official docs |

## Usage

AI assistant sẽ tự động load skill phù hợp dựa trên context. Để chỉ định cụ thể:

```
Use the vn-debugging-cv skill to analyze these OCR errors:
- Expected: "12A-12345"
- Got: "12A-123S5"
```

## Project Structure Reference

```
ComputerVisionNew/
├── src/
│   ├── io/              # Input adapters
│   ├── detector/        # YOLO, DeepSolo
│   ├── preprocess/      # Crop, rectify, denoise
│   ├── ocr/            # EasyOCR, TrOCR
│   ├── postprocess/     # Regex, normalization
│   ├── pipeline/        # End-to-end orchestration
│   ├── eval/           # CER, WER, accuracy
│   └── app/            # CLI, demo
├── scripts/             # Training, inference scripts
├── data/               # Manifests, GT, test sets
├── configs/            # Model configs
├── weights/            # Trained models (.gitkeep)
├── reports/            # Evaluation reports
└── docs/               # Notebooks, thesis docs
```

## Quick Commands

```bash
# Inference
python scripts/run_infer.py --manifest data/test.csv --output-json out.json

# Evaluation
python scripts/eval_pipeline.py --pred-csv outputs/pred.csv

# Error Analysis
python scripts/export_char_errors_csv.py --pred-csv out.csv --output-csv errors.csv

# Compare Methods
python scripts/run_buoi4_experiments.py --config-a-csv a.csv --config-b-csv b.csv
```

## Standard Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| CER | < 0.05 | Character Error Rate |
| WER | < 0.10 | Word Error Rate |
| Accuracy | > 90% | Plate-level exact match |
| Latency | < 100ms | Mean inference time |

## Vietnam License Plate Format

- Format: `XXY-NNNNN`
- XX: Province code (2 digits)
- Y: Letter (A-Z)
- NNNNN: Serial (5 digits)
- Examples: `12A-12345`, `43K-98765`

## Skills Development

Skills được customize từ [agent-skills](https://github.com/toanlion24/agent-skills) của Addison Omar.
