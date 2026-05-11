# Vietnam License Plate Recognition

Skeleton architecture for a modular VN license plate pipeline:

- `src/io`: input readers.
- `src/detector`: plate detector wrappers.
- `src/preprocess`: crop and image preprocessing ops.
- `src/ocr`: OCR wrappers.
- `src/postprocess`: regex and repair rules.
- `src/pipeline`: end-to-end orchestration.
- `src/eval`: CER/WER/plate accuracy and exports.
- `src/app`: CLI/demo entry logic.
- `scripts`: train, infer, manifest, evaluate entrypoints.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run batch inference with real YOLOv8 + EasyOCR:

```bash
python scripts/run_infer.py --input-dir data/raw --output-json outputs/predictions.json --detector-backend yolov8 --detector-model weights/yolov8_license_plate.pt --ocr-backend easyocr
```

Run batch inference with YOLOv8 + TrOCR:

```bash
python scripts/run_infer.py --input-dir data/raw --output-json outputs/predictions_trocr.json --detector-backend yolov8 --detector-model weights/yolov8_license_plate.pt --ocr-backend trocr --trocr-model microsoft/trocr-base-printed --device cpu
```

Run with CUDA (auto-fallback to CPU if CUDA unavailable):

```bash
python scripts/run_infer.py --input-dir data/raw --output-json outputs/predictions_trocr_cuda.json --detector-backend yolov8 --detector-model weights/yolov8_license_plate.pt --ocr-backend trocr --device cuda
```

Build a manifest:

```bash
python scripts/build_manifest.py --input-dir data/raw --output-csv data/manifest.csv
```

Create train/val/test splits:

```bash
python scripts/split_dataset.py --input-dir data/raw --output-dir data/splits --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42
```

Run quick EDA for images + YOLO labels:

```bash
python scripts/eda_dataset.py --images-dir data/raw --labels-dir data/labels --output-dir reports/eda --num-preview 12
```

Create basic preprocess/augment variants:

```bash
python scripts/preprocess_augment.py --input-dir data/raw --output-dir data/interim/augmented --size 640 640 --seed 42
```

Evaluate predictions:

```bash
python scripts/eval_pipeline.py --pred-csv reports/pred_vs_gt.csv
```

Phân tích lỗi theo **ký tự + vùng** (tỉnh / chữ / serial) từ CSV prediction (Buổi 4):

```bash
python scripts/export_char_errors_csv.py --pred-csv outputs/buoi4/deepsolo_e2e_predictions.csv --output-csv reports/char_errors_by_region.csv
```

## Dữ liệu tổng hợp + chạy metric có số thật (không cần ảnh ngoài Git)

Sinh ảnh biển + `data/test_manifest.csv`, rồi đánh giá A/B (EasyOCR vs TrOCR) với inference thật:

```bash
python scripts/generate_synthetic_plate_dataset.py
python scripts/run_buoi4_manifest_inference.py --manifest data/test_manifest.csv --detector-backend dummy --device cpu --run-metrics --metrics-json reports/buoi4_ab_metrics.json --report-md reports/buoi4_ab_run_synthetic.md
```

Chi tiết và ý nghĩa báo cáo: [`data/synthetic_plates/README.md`](data/synthetic_plates/README.md).

(Lần đầu TrOCR/EasyOCR có thể tải model; detector `dummy` chỉ để lấy vùng crop trung tâm — với ảnh thật nên dùng `yolov8`.)

Ghép `data/test_manifest.csv` từ thư mục ảnh + CSV/JSON hoặc file `.txt` nhãn: [`scripts/build_test_manifest_from_folder.py`](scripts/build_test_manifest_from_folder.py) — xem [`data/manifests/README.md`](data/manifests/README.md).

## Đối chiếu đề tài, Buổi 4 đầy đủ, và bảo vệ

- **Checklist + ma trận yêu cầu + câu hỏi hội đồng:** [`docs/KIEM_TRA_DE_TAI_VA_HOI_DONG.md`](docs/KIEM_TRA_DE_TAI_VA_HOI_DONG.md)
- **Kiểm tra nhanh** (compile, weight, manifest, link tới tài liệu trên):

```bash
python scripts/check_de_tai_readiness.py
```

- **Đánh giá A/B có manifest + GT** (YOLOv8 + EasyOCR vs YOLOv8 + TrOCR; có thể nhập CSV DeepSolo thay cho một hoặc hai nhánh): `scripts/run_buoi4_manifest_inference.py` — xem docstring script.
- **Fine-tune TrOCR:** `scripts/train_trocr.py` và `configs/trocr/finetune_defaults.json`.

Run Buổi 4 A/B comparison after exporting both prediction CSV files:

```bash
python scripts/run_buoi4_experiments.py --config-a-csv outputs/buoi4/deepsolo_e2e_predictions.csv --config-b-csv outputs/buoi4/deepsolo_trocr_predictions.csv
```

Create a small Buổi 4 demo result in `docs/`:

```bash
python scripts/create_buoi4_demo_predictions.py --output-dir outputs/buoi4/demo
python scripts/run_buoi4_experiments.py --config-a-csv outputs/buoi4/demo/deepsolo_e2e_predictions.csv --config-b-csv outputs/buoi4/demo/deepsolo_trocr_predictions.csv --metrics-json reports/buoi4_demo_ab_metrics.json --report-md docs/buoi-4-ket-qua-thuc-nghiem-deepsolo-trocr.md
```

Prepare DeepSolo-style annotations from a Buổi 4 manifest:

```bash
python scripts/prepare_buoi4_deepsolo_data.py --manifest-csv data/manifests/buoi4_test.csv --output-dir data/deepsolo/buoi4 --split test
```

## Notes

- `--detector-backend dummy` and `--ocr-backend dummy` remain available for wiring tests.
- Default detector weight path is `weights/yolov8_license_plate.pt`; update it to your trained model.
- TrOCR models are auto-downloaded from Hugging Face on first run and cached locally.
- You can override cache path with `--model-cache-dir`.

## Workflow notebook + Git

- Dùng `.py` trong `src/` và `scripts/` làm code chính.
- Dùng `.ipynb` trong `docs/` chỉ để demo, trình bày và EDA.
- Checklist ngắn trước commit: `docs/pre-commit-checklist.md`.
- Trước khi commit notebook, xóa output để file nhẹ:

```bash
python scripts/strip_ipynb_outputs.py docs/buoi-2-eda-visual.ipynb
```

- Nếu cần, bạn có thể clean nhiều notebook cùng lúc:

```bash
python scripts/strip_ipynb_outputs.py docs/*.ipynb
```

