"""Quick test: Check what detector sees in user's image."""
import sys
sys.path.insert(0, "D:/ComputerVisionNew")

from PIL import Image
import numpy as np
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.utils.types import FrameData

print("Loading detector...")
detector = YoloV8PlateDetector("runs/detect/yolo_cropped_v2/weights/best.pt", conf_threshold=0.05)

# Test on sample
img = Image.open("data/samples/sample_1_30G12345.jpg").convert('RGB')
frame = np.array(img)

dets = detector.predict(FrameData("test", frame, "test"))
print(f"\nSample image: {len(dets)} detections")
for d in dets:
    print(f"  Score: {d.score:.3f}, BBox: {d.bbox_xyxy}")

# Instructions
print("\n" + "="*50)
print("TO TEST YOUR IMAGE:")
print("1. Lower confidence threshold in sidebar to 0.05")
print("2. Enable 'Enable OCR' checkbox")
print("3. Upload your image")
print("="*50)
