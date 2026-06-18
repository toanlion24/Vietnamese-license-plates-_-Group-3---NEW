"""Qwen2-VL adapter for VN license plate OCR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.postprocess.plate_rules import normalize_plate_text
from src.utils.types import OcrResult, PlateCrop

try:
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info
    import peft
    from transformers import BitsAndBytesConfig
except ImportError:
    Qwen2VLForConditionalGeneration = None
    Qwen2VLProcessor = None
    torch = None
    Image = None
    peft = None
    BitsAndBytesConfig = None


SYSTEM_PROMPT = """Bạn là một hệ thống nhận diện biển số xe Việt Nam.
Đọc biển số xe trong ảnh và chỉ trả về kết quả, không giải thích.
Định dạng: [mã tỉnh][chữ cái loại][số]
Ví dụ: 30G112345

Nếu không đọc được, trả về: UNREADABLE"""


@dataclass
class QwenVLHypothesis:
    text: str
    score: float


class Qwen2VLPlateOcr:
    """Qwen2-VL adapter for VN license plate recognition.

    Supports both full model and LoRA adapter loading.
    """

    def __init__(
        self,
        model_name: str = "Qwen2VL-2B-Instruct",
        device: str = "auto",
        cache_dir: str | None = None,
        max_new_tokens: int = 32,
        temperature: float = 0.1,
        use_lora_adapter: bool = True,
        use_bnb_quant: bool = True,
    ) -> None:
        if Qwen2VLForConditionalGeneration is None:
            raise ImportError(
                "transformers, peft, and qwen_vl_utils are required. "
                "Install with: pip install transformers peft qwen-vl-utils torch"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logging.info("Loading Qwen2-VL model: %s", model_name)
        
        # Default base model name
        base_model_name = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"

        if use_lora_adapter:
            logging.info("Loading Qwen2-VL with LoRA adapter: %s", model_name)
            
            # Check if LoRA adapter exists
            adapter_path = Path(model_name)
            if not (adapter_path / "adapter_config.json").exists():
                nested = adapter_path / "qwen_vl_finetuned"
                if (nested / "adapter_config.json").exists():
                    adapter_path = nested
            
            lora_exists = (adapter_path / "adapter_config.json").exists()
            
            if lora_exists and use_lora_adapter and device == "cuda":
                try:
                    # Load base model with 4-bit quantization (GPU only)
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model_name,
                        cache_dir=cache_dir,
                        device_map=device,
                        quantization_config=bnb_config,
                    )
                    
                    # Load LoRA adapter
                    from peft import PeftModel
                    logging.info("Loading LoRA adapter from: %s", adapter_path)
                    self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
                    self.model.eval()
                    
                    # Processor from base model
                    self.processor = Qwen2VLProcessor.from_pretrained(base_model_name)
                    logging.info("LoRA adapter loaded successfully")
                except Exception as e:
                    logging.warning("Failed to load LoRA adapter: %s. Falling back to base model.", str(e))
                    use_lora_adapter = False
            else:
                # CPU or no LoRA: Use base model without quantization
                logging.info("Using base model (LoRA disabled on CPU)")
                use_lora_adapter = False
        
        if not use_lora_adapter:
            # Load base model without LoRA
            # For CPU: use non-quantized model (Qwen2-VL-2B-Instruct)
            # For GPU: can use quantized model
            if device == "cuda":
                model_to_use = base_model_name  # unsloth 4bit model
                use_quantization = True
            else:
                # CPU: use original Qwen2-VL model (not quantized)
                model_to_use = "Qwen/Qwen2-VL-2B-Instruct"
                use_quantization = False
            
            logging.info("Loading base model: %s", model_to_use)
            
            if use_quantization and device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_to_use,
                        cache_dir=cache_dir,
                        device_map=device,
                        quantization_config=bnb_config,
                    )
                except Exception as e:
                    logging.warning("Quantization failed: %s", str(e))
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_to_use,
                        cache_dir=cache_dir,
                        torch_dtype=torch.bfloat16,
                        device_map=device,
                    )
            else:
                # No quantization (CPU or fallback)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_to_use,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32,
                    device_map=device,
                )
            
            self.processor = Qwen2VLProcessor.from_pretrained(model_to_use, cache_dir=cache_dir)
            self.model.eval()
            logging.info("Base model loaded successfully")

        self.model.eval()
        logging.info("Qwen2-VL model loaded successfully")

    def _preprocess_crop(self, crop: Image.Image) -> Image.Image:
        """Preprocess plate crop for Qwen2-VL."""
        w, h = crop.size
        
        # Resize to reasonable size while keeping aspect ratio
        target_size = 448
        if max(w, h) > target_size:
            ratio = target_size / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            crop = crop.resize((new_w, new_h), Image.LANCZOS)
        
        return crop

    def recognize(self, plate_crop: PlateCrop, preprocessed=None) -> OcrResult:
        """Recognize plate text from crop image."""
        import numpy as np

        # Convert numpy array to PIL Image if needed
        if preprocessed is not None:
            if isinstance(preprocessed, np.ndarray):
                if len(preprocessed.shape) == 2:
                    pil_img = Image.fromarray(preprocessed, mode="L").convert("RGB")
                else:
                    pil_img = Image.fromarray(preprocessed)
            else:
                pil_img = preprocessed
        else:
            pil_img = Image.fromarray(plate_crop.crop)

        pil_img = self._preprocess_crop(pil_img)

        # Build conversation
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_img,
                    },
                    {"type": "text", "text": "Đọc biển số xe trong ảnh này:"},
                ],
            },
        ]

        text = self._generate(conversation)
        text_raw = text.strip()
        text_norm = normalize_plate_text(text_raw)

        return OcrResult(
            image_id=plate_crop.image_id,
            text_raw=text_raw,
            text_norm=text_norm,
            ocr_score=1.0,  # Qwen2-VL doesn't provide confidence by default
        )

    def _generate(self, conversation: list[dict]) -> str:
        """Generate text from conversation."""
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def recognize_with_hypotheses(
        self, plate_crop: PlateCrop, preprocessed=None
    ) -> tuple[OcrResult, list[QwenVLHypothesis]]:
        """Return result with multiple hypotheses using temperature sampling."""
        import numpy as np

        if preprocessed is not None:
            if isinstance(preprocessed, np.ndarray):
                if len(preprocessed.shape) == 2:
                    pil_img = Image.fromarray(preprocessed, mode="L").convert("RGB")
                else:
                    pil_img = Image.fromarray(preprocessed)
            else:
                pil_img = preprocessed
        else:
            pil_img = Image.fromarray(plate_crop.crop)

        pil_img = self._preprocess_crop(pil_img)

        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_img,
                    },
                    {"type": "text", "text": "Đọc biển số xe trong ảnh này:"},
                ],
            },
        ]

        text = self._generate(conversation)
        text_norm = normalize_plate_text(text)

        result = OcrResult(
            image_id=plate_crop.image_id,
            text_raw=text,
            text_norm=text_norm,
            ocr_score=1.0,
        )

        hypothesis = QwenVLHypothesis(text=text, score=1.0)

        return result, [hypothesis]


# Backward compatibility
QwenVLPlateAdapter = Qwen2VLPlateOcr
