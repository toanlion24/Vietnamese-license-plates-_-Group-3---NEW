"""
Merge LoRA adapter with base model and save as a full model.
Run this once to create a merged model that can be loaded without bitsandbytes.
Saves to C:/temp_qwen2vl_merged/ to avoid disk space issues on D: drive.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Output to C: drive (has ~75GB free)
    output_dir = Path("c:/temp_qwen2vl_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect adapter path (may be nested)
    adapter_path = Path("d:/ComputerVisionNew/experiments/qwen2vl_finetuned")
    if not (adapter_path / "adapter_config.json").exists():
        nested = adapter_path / "qwen_vl_finetuned"
        if (nested / "adapter_config.json").exists():
            adapter_path = nested

    logger.info("Step 1: Loading base model (Qwen/Qwen2-VL-2B-Instruct)...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    logger.info("Base model loaded. Params: %.1fB", sum(p.numel() for p in base_model.parameters()) / 1e9)

    logger.info("Step 2: Loading LoRA adapter from %s...", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    logger.info("LoRA adapter loaded")

    logger.info("Step 3: Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    logger.info("Merge complete. Merged model params: %.1fB", sum(p.numel() for p in model.parameters()) / 1e9)

    logger.info("Step 4: Saving merged model to %s...", output_dir)
    model.save_pretrained(str(output_dir))
    logger.info("Model saved successfully")

    logger.info("Step 5: Saving processor...")
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    processor.save_pretrained(str(output_dir))
    logger.info("Processor saved successfully")

    logger.info("=" * 60)
    logger.info("DONE! Merged model saved to: %s", output_dir)
    logger.info("You can now load it with:")
    logger.info("  Qwen2VLForConditionalGeneration.from_pretrained('%s')", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
