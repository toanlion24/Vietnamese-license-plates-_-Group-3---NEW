"""
Convert merged model to float16 to reduce memory usage for CPU inference.
"""
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

MERGED_DIR = Path("c:/temp_qwen2vl_merged")
OUTPUT_DIR = Path("c:/temp_qwen2vl_merged_fp16")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading merged model (float32)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MERGED_DIR),
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("Converting to float16...")
    model = model.to(dtype=torch.float16)
    print("Saving float16 model...")
    model.save_pretrained(str(OUTPUT_DIR))
    print("Saving processor...")
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"DONE! Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
