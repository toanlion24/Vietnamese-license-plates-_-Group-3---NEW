"""Compare all YOLO models on sample image."""
import sys
sys.path.insert(0, "D:/ComputerVisionNew")

from PIL import Image
import numpy as np
from ultralytics import YOLO
from src.utils.types import FrameData

# Sample image
img_path = "data/samples/sample_1_30G12345.jpg"
img = Image.open(img_path).convert('RGB')
frame = np.array(img)

models = [
    "runs/detect/experiments/detector/yolov8n_augmented/weights/best.pt",
    "runs/detect/experiments/detector/yolov8n_crops/weights/best.pt",
    "runs/detect/experiments/detector/yolov8n_finetune/weights/best.pt",
    "runs/detect/yolo_augmented_v1/weights/best.pt",
    "runs/detect/yolo_cropped_v2/weights/best.pt",
    "runs/detect/yolo_full_images/weights/best.pt",
]

for model_path in models:
    import os
    if not os.path.exists(model_path):
        print(f"SKIP (not found): {model_path}")
        continue
        
    print(f"\n=== {model_path} ===")
    model = YOLO(model_path)
    
    # Check model info
    print(f"Model classes: {model.names}")
    
    # Test with various conf thresholds
    for conf in [0.01, 0.05, 0.10, 0.25, 0.50]:
        results = model.predict(frame, conf=conf, verbose=False)
        count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
        if count > 0:
            print(f"  conf={conf}: {count} detections")
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        print(f"    bbox={box.xyxy[0].tolist()}, conf={box.conf[0]:.3f}")
