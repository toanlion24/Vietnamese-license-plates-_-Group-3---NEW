# Vietnam License Plate Recognition

Hệ thống nhận diện biển số xe Việt Nam sử dụng **YOLOv8n** (Detection) và **Qwen2-VL-2B-Instruct** (OCR/VLM).

## Công nghệ

| Thành phần | Công nghệ |
|------------|-----------|
| Detection | YOLOv8 Nano (YOLOv8n) |
| OCR/VLM | Qwen2-VL-2B-Instruct (fine-tuned) |
| Fine-tuning | Unsloth + QLoRA (4-bit) |
| UI | Streamlit |

## Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  DETECTION  │───▶│  CROPPING   │───▶│   OCR/VLM   │───▶│ POST-PROCESS│
│  YOLOv8n    │    │  Auto-crop  │    │ Qwen2-VL-2B │    │ Regex+Rules │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Cấu trúc dự án

```
src/
  io/              # Đọc ảnh, video, webcam
  detector/        # YOLOv8 detector adapter
  ocr/             # Qwen2-VL adapter
  preprocess/      # Tiền xử lý ảnh
  postprocess/     # Hậu xử lý regex/luật biển số VN
  pipeline/        # Pipeline tổng hợp
  eval/            # Metrics (CER, WER, plate accuracy)
  app/             # Demo CLI/Streamlit
  utils/           # Utilities
scripts/
  train_yolo.py    # Train YOLO detector
  train_qwen.py    # Fine-tune Qwen2-VL (chạy trên Colab)
  run_inference.py # Inference batch
  eval_pipeline.py # Đánh giá pipeline
configs/
  yolo/            # Cấu hình YOLO
  qwen_vl/         # Cấu hình QLoRA, prompt
data/
  raw/             # Ảnh gốc
  labels/          # Nhãn YOLO
  splits/          # Train/val/test splits
  manifests/       # Manifest cho OCR training
experiments/
  yolo/            # Checkpoint YOLO
  qwen_vl/         # Checkpoint Qwen2-VL fine-tuned
reports/          # Báo cáo từng buổi
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train YOLO Detector

```bash
python scripts/train_yolo.py --data data/yolo_dataset_full/data.yaml --epochs 50 --batch 16
```

### 2. Fine-tune Qwen2-VL (Google Colab)

Xem notebook: `notebooks/train_qwen2vl_finetune.ipynb`

```python
# Cài đặt Unsloth
!pip install unsloth unsloth_granite

from unsloth import FastVisionModel

# Load model
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

# Apply LoRA
model = FastVisionModel.get_peft_model(
    model,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Train và push lên Hugging Face
# model.push_to_hub_LoRA("username/vn-plate-qwen2-vl-2b")
```

### 3. Inference

```bash
python scripts/run_inference.py \
  --input-dir data/test_images \
  --detector-model weights/yolov8n_plate.pt \
  --qwen-model username/vn-plate-qwen2-vl-2b \
  --output-json outputs/predictions.json
```

### 4. Đánh giá

```bash
python scripts/eval_pipeline.py --pred-csv outputs/predictions.csv
```

## Metrics

- **CER** (Character Error Rate): Tổng edit distance ký tự / tổng ký tự GT
- **WER** (Word Error Rate): Edit distance token / tổng token GT
- **Plate-level Accuracy**: Tỉ lệ pred == gt sau chuẩn hóa

## Ghi chú

- Model Qwen2-VL-2B-Instruct được fine-tune với Unsloth để tối ưu VRAM
- Sử dụng QLoRA (4-bit quantization) để giảm kích thước model
- Pipeline hỗ trợ input từ ảnh, video, hoặc webcam realtime
