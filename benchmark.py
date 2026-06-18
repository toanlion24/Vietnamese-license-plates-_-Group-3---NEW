"""Benchmark: What takes how long?"""
import sys
import time
sys.path.insert(0, "D:/ComputerVisionNew")

from PIL import Image
import numpy as np
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.ocr.qwen_adapter import Qwen2VLPlateOcr
from src.utils.types import FrameData

# Load models
print("Loading models...")
t0 = time.time()
detector = YoloV8PlateDetector("runs/detect/yolo_cropped_v2/weights/best.pt", conf_threshold=0.15)
print(f"Detector loaded: {time.time()-t0:.1f}s")

t0 = time.time()
ocr = Qwen2VLPlateOcr("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit", device="cpu", use_lora_adapter=False)
print(f"OCR loaded: {time.time()-t0:.1f}s")

# Test
img = Image.open("data/samples/sample_1_30G12345.jpg").convert('RGB')
frame = np.array(img)
frame_data = FrameData("test", frame, "test")

print("\n--- BENCHMARK ---")
t0 = time.time()
dets = detector.predict(frame_data)
print(f"Detection: {time.time()-t0*1000:.0f}ms")

if dets:
    best = dets[0]
    t0 = time.time()
    ocr_out = ocr.recognize(best, best.crop)
    print(f"OCR: {time.time()-t0*1000:.0f}ms")

print("\n--- SOLUTIONS ---")
print("1. GPU: OCR ~500ms (10x faster)")
print("2. TrOCR: ~100ms on CPU (smaller model)")
print("3. EasyOCR: ~50ms on CPU (lightweight)")
print("4. Detection only: ~50ms total")
