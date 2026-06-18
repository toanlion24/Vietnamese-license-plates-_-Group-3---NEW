---
name: vn-model-training
description: Training và fine-tuning models cho VN license plate recognition. Dùng khi fine-tune Qwen2-VL, train YOLO detector, hoặc tạo synthetic dataset.
---

# VN License Plate - Model Training

## Overview

Training workflow cho VN plate models với **Qwen2-VL-2B-Instruct** (OCR) và **YOLOv8n** (Detection).

## Công nghệ

| Thành phần | Tool |
|------------|------|
| Base Model | Qwen2-VL-2B-Instruct |
| Fine-tuning | Unsloth |
| Quantization | QLoRA (4-bit) |
| Platform | Google Colab |

## When to Use

- Fine-tune Qwen2-VL cho VN plate OCR
- Train YOLO detector cho plate localization
- Tạo synthetic dataset
- Evaluate model performance

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATA ──→ 2. PREPARE ──→ 3. TRAIN ──→ 4. EVALUATE ──┐  │
│      │              │             │              │       │  │
│      │              │             │              ▼       │  │
│      │              │             │         5. DEPLOY    │  │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Data Preparation

### Manifest Structure

```python
# Required columns for manifest
manifest = {
    "image_id": "001",           # Unique ID
    "image_path": "data/raw/001.jpg",
    "plate_text": "30G112345",   # Ground truth
    "split": "train/val/test"    # Dataset split
}
```

### Crop cho Qwen2-VL Training

```bash
# Export crops từ detector output
python scripts/export_qwen_crops.py \
    --manifest data/manifests/train.csv \
    --detector-model weights/yolov8n_plate.pt \
    --output-dir data/crops/train
```

### Data Format cho Qwen2-VL

```json
{
  "messages": [
    {"role": "user", "content": [{"type": "image"}, "Đọc biển số xe trong ảnh"]}, 
    {"role": "assistant", "content": "30G112345"}
  ]
}
```

## Step 2: Prepare Training Data

### YOLO Format

```bash
# Convert to YOLO format
python scripts/reorganize_yolo_data.py \
    --manifest data/manifests/train.csv \
    --output-dir data/yolo \
    --image-size 640

# Structure:
# data/yolo/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── labels/
#       ├── train/
#       └── val/
```

## Step 3: Training

### Qwen2-VL Fine-tuning với Unsloth

**Chạy trên Google Colab**

```python
# Cài đặt Unsloth
!pip install unsloth unsloth_granite

from unsloth import FastVisionModel
import torch

# Load model với 4-bit quantization
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

# Apply LoRA
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing="unsloth"
)

# Prepare dataset (conversation format)
from datasets import load_dataset

def format_conversations(examples):
    conversations = []
    for image, text in zip(examples["image"], examples["plate_text"]):
        conv = [
            {"role": "user", "content": [
                {"type": "image"},
                "Đọc biển số xe trong ảnh"
            ]},
            {"role": "assistant", "content": text}
        ]
        conversations.append({"messages": conv})
    return {"conversations": conversations}

dataset = load_dataset("path/to/your/dataset")
dataset = dataset.map(format_conversations, batched=True)

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="conversations",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs/qwen_vl_finetuned",
        report_to="none",
    ),
)

trainer.train()

# Push lên Hugging Face
model.push_to_hub_LoRA("username/vn-plate-qwen2-vl-2b")
```

### QLoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| r | 16 | LoRA rank |
| lora_alpha | 32 | Scaling factor |
| target_modules | all linear | Modules to adapt |
| bias | none | Don't train biases |
| load_in_4bit | True | 4-bit quantization |

### YOLO Training

```bash
# Train detector
python scripts/train_detector.py \
    --model yolov8n.pt \
    --data data/yolo/plates.yaml \
    --epochs 50 \
    --image-size 640 \
    --device cuda

# Resume from checkpoint
python scripts/train_detector.py \
    --model outputs/detector/train/weights/last.pt \
    --data data/yolo/plates.yaml \
    --resume
```

### Training Best Practices

```python
# Always set seeds for reproducibility
def set_training_seed(seed=42):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Monitor GPU memory
import gc
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Step 4: Evaluation

### Standard Metrics

```bash
# Run full evaluation
python scripts/eval_pipeline.py \
    --pred-csv outputs/predictions.csv \
    --gt-csv data/gt.csv

# Output:
# {
#   "cer": 0.023,      # Character Error Rate (lower is better)
#   "wer": 0.087,      # Word Error Rate
#   "accuracy": 0.913,  # Plate-level accuracy
#   "mean_latency_ms": 45.2
# }
```

### Per-Stage Evaluation

```python
# Evaluate detector separately
from src.detector import YoloV8PlateDetector

detector = YoloV8PlateDetector("weights/yolov8n_plate.pt")
detections = detector.predict(frame_data)

# Evaluate OCR separately
from src.ocr import Qwen2VLPlateOcr

ocr = Qwen2VLPlateOcr("username/vn-plate-qwen2-vl-2b")
result = ocr.recognize(plate_crop, pil_image)
```

## Step 5: Deployment

### Inference

```python
from src.pipeline import PlateInferencePipeline
from src.detector import YoloV8PlateDetector
from src.ocr import Qwen2VLPlateOcr

# Load models
detector = YoloV8PlateDetector("weights/yolov8n_plate.pt")
ocr = Qwen2VLPlateOcr("username/vn-plate-qwen2-vl-2b")

# Create pipeline
pipeline = PlateInferencePipeline(detector, ocr)

# Run inference
result = pipeline.run(frame_data)
print(f"Plate: {result.plate_text}, Confidence: {result.confidence:.2f}")
```

### Model Card Template

```markdown
# Model Card: VN License Plate Qwen2-VL OCR v1.0

## Model Details
- Type: Qwen2-VL-2B-Instruct fine-tuned với Unsloth + QLoRA
- Input: Plate crop images (variable size)
- Output: Vietnamese license plate text

## Training Data
- Real crops: 200
- Training epochs: 3
- Fine-tuning method: QLoRA (4-bit)

## Performance
| Metric | Value |
|--------|-------|
| CER | 0.023 |
| WER | 0.087 |
| Plate Accuracy | 91.3% |

## Usage
```python
from src.ocr import Qwen2VLPlateOcr
ocr = Qwen2VLPlateOcr("username/vn-plate-qwen2-vl-2b")
result = ocr.recognize(plate_crop, pil_image)
```
```

## Red Flags

- Training without validation set
- No seed setting
- Ignoring overfitting signs
- Evaluating on training data
- No error analysis after training
- Hard-coding model paths

## Verification Checklist

After training:

- [ ] Validation metrics measured
- [ ] No overfitting (val loss close to train loss)
- [ ] Error analysis done
- [ ] Model pushed to Hugging Face
- [ ] Inference script tested
