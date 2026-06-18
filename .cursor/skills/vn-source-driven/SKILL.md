---
name: vn-source-driven
description: Ground framework decisions in official documentation. Dùng khi implement features với YOLOv8, Qwen2-VL, Unsloth, hoặc các thư viện ML khác.
---

# VN License Plate - Source-Driven Development

## Overview

Verify framework decisions against official documentation. Cite sources, flag unverified claims, ensure code matches best practices từ library creators.

## When to Use

- Using any ML/DL library (YOLOv8, Qwen2-VL, Unsloth, PyTorch, etc.)
- Implementing model training or inference
- Configuring hyperparameters
- Troubleshooting model behavior

## The Source Verification Process

```
┌─────────────────────────────────────────────────┐
│  1. FIND SOURCE ──→ 2. VERIFY ──→ 3. IMPLEMENT │
│         │                 │               │      │
│         ▼                 ▼               ▼      │
│   Official docs      Check against     Cite in   │
│   GitHub, paper     source            code      │
└─────────────────────────────────────────────────┘
```

## Common Sources

### YOLOv8 (Ultralytics)

| Topic | Source |
|-------|--------|
| Training | https://docs.ultralytics.com/modes/train/ |
| Inference | https://docs.ultralytics.com/modes/predict/ |
| Export | https://docs.ultralytics.com/modes/export/ |
| CLI | https://docs.ultralytics.com/cli/ |

### Qwen2-VL (Alibaba)

| Topic | Source |
|-------|--------|
| Hugging Face | https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct |
| GitHub | https://github.com/QwenLM/Qwen2-VL |
| Paper | https://arxiv.org/abs/2409.12191 |

### Unsloth

| Topic | Source |
|-------|--------|
| GitHub | https://github.com/unslothai/unsloth |
| Documentation | https://docs.unsloth.ai/ |
| Blog | https://unsloth.ai/blog |

### PyTorch

| Topic | Source |
|-------|--------|
| Tutorials | https://pytorch.org/tutorials/ |
| Docs | https://pytorch.org/docs/ |
| Recipes | https://pytorch.org/tutorials/recipes/recipes_index.html |

## Implementation Examples

### YOLOv8 Training (Verified)

```python
# VERIFIED: From https://docs.ultralytics.com/modes/train/
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # nano model (fastest)

# Train with verified parameters
results = model.train(
    data='data/yolo/plates.yaml',  # Dataset config
    epochs=50,
    imgsz=640,                      # Image size
    batch=16,                       # Batch size
    device=0,                       # GPU device
    patience=10,                    # Early stopping
    save=True,                      # Save checkpoints
    project='outputs/detector',     # Save location
    name='train',                   # Experiment name
)
```

### Qwen2-VL Inference (Verified)

```python
# VERIFIED: From https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

# Load model and processor
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Build conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Đọc biển số xe trong ảnh:"},
        ],
    }
]

# Apply chat template
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(conversation)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# Generate
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=32)

output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Unsloth Fine-tuning (Verified)

```python
# VERIFIED: From https://docs.unsloth.ai/
from unsloth import FastVisionModel

# Load model with 4-bit quantization
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

# Train
model.fit(
    lr=2e-4,
    epochs=3,
    batch_size=2,
    grad_accumulation=4,
)

# Push to Hugging Face
model.push_to_hub_LoRA("username/vn-plate-qwen2-vl-2b")
```

### QLoRA Configuration (Verified)

```python
# VERIFIED: From https://arxiv.org/abs/2306.12967 (QLoRA paper)
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # Rank (higher = more params, more memory)
    lora_alpha=32,           # Scaling factor (usually 2x rank)
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,       # Dropout for regularization
    bias="none",             # Don't train biases (saves memory)
    task_type="CAUSAL_LM",   # For language models
)
```

## Source Citation Format

```python
def recognize_plate(crop):
    """
    VN plate recognition using Qwen2-VL.
    
    Source: Qwen2-VL paper (https://arxiv.org/abs/2409.12191)
    Implementation: Hugging Face transformers + qwen_vl_utils
    
    Args:
        crop: Preprocessed plate crop (PIL Image)
    
    Returns:
        str: Recognized plate text
    
    Note: Resize to 448-896px for optimal Qwen2-VL performance
    """
    # ...
```

## Unverified Claims to Flag

| Claim | Status | Source Needed |
|-------|--------|--------------|
| "Qwen2-VL is better than TrOCR" | ❌ UNVERIFIED | Needs experiment |
| "4-bit quantization is lossless" | ⚠️ PARTIAL | Depends on model, QLoRA paper suggests minimal loss |
| "YOLOv8 mAP > 0.9 is good" | ✅ VERIFIED | Industry standard |
| "LoRA r=16 is optimal" | ❌ UNVERIFIED | Depends on dataset size |

## Common Mistakes (Verified)

### ❌ BAD: Hard-coded without source

```python
# BAD: Why batch_size=32? No source
model.train(batch_size=32)
```

### ✅ GOOD: Sourced and justified

```python
# GOOD: batch_size=16 from YOLO docs for 8GB GPU
# Source: https://docs.ultralytics.com/modes/train/
model.train(batch_size=16, imgsz=640)
```

### ❌ BAD: Assumed optimal parameters

```python
# BAD: Learning rate assumption
model.train(lr=0.001)  # No justification
```

### ✅ GOOD: From official recipe

```python
# GOOD: Learning rate from Unsloth docs for fine-tuning
# Source: https://docs.unsloth.ai/
model.fit(lr=2e-4, epochs=3)  # From Unsloth recommendations
```

## Verification Checklist

```markdown
## Source Verification

### Implementation: [What you're implementing]

### Source Found:
- [ ] Official documentation (link)
- [ ] GitHub example (link)
- [ ] Paper citation (if applicable)

### Verification:
- [ ] Parameters match source
- [ ] API usage correct
- [ ] Dependencies documented

### Deviations from Source:
- [ ] None
- [ ] Documented reasons for changes

### Flagged for Testing:
- [ ] Any unverified claims
- [ ] Domain-specific optimizations
```

## Red Flags

- No source for ML hyperparameters
- Using parameters without justification
- Assuming default = optimal
- Ignoring official recommendations
- Implementing "common knowledge" without verification

## Resources

### Official Repositories

- YOLOv8: https://github.com/ultralytics/ultralytics
- Qwen2-VL: https://github.com/QwenLM/Qwen2-VL
- Unsloth: https://github.com/unslothai/unsloth
- Transformers: https://github.com/huggingface/transformers

### Papers

- QLoRA: https://arxiv.org/abs/2306.12967
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- YOLOv8: https://docs.ultralytics.com/
