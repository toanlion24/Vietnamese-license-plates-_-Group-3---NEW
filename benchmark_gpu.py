"""Benchmark OCR on GPU."""
import sys
import time
import numpy as np
sys.path.insert(0, "D:/ComputerVisionNew")

from PIL import Image
from src.ocr.qwen_adapter import Qwen2VLPlateOcr
from src.utils.types import PlateCrop

# Load OCR on GPU
print("Loading Qwen2-VL on GPU...")
t0 = time.time()
ocr = Qwen2VLPlateOcr('unsloth/Qwen2-VL-2B-Instruct-bnb-4bit', device='cuda', use_lora_adapter=False)
print(f"Loaded: {time.time()-t0:.1f}s\n")

# Load test image
img = Image.open("data/samples/sample_1_30G12345.jpg").convert('RGB')
crop = np.array(img)[195:286, 164:478]
plate_crop = PlateCrop("test", crop, (0,0,314,91), 0.8)

# Benchmark
print("Running OCR benchmark...")
times = []
for i in range(3):
    t0 = time.time()
    result = ocr.recognize(plate_crop, crop)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed*1000:.0f}ms -> '{result.text_raw}'")

print(f"\nAverage: {np.mean(times)*1000:.0f}ms")
print(f"Min: {min(times)*1000:.0f}ms")
