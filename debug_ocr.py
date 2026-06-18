"""Test OCR on real plate image."""
import sys
sys.path.insert(0, "D:/ComputerVisionNew")

from PIL import Image
import numpy as np
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.ocr.qwen_adapter import Qwen2VLPlateOcr
from src.preprocess.ops import crop_plate, preprocess_plate
from src.postprocess.plate_rules import normalize_plate_text, advanced_repair_ocr_text
from src.utils.types import FrameData

# Load detector
print("Loading detector...")
yolo_path = "runs/detect/yolo_cropped_v2/weights/best.pt"
detector = YoloV8PlateDetector(yolo_path, conf_threshold=0.15)

# Load OCR
print("Loading OCR...")
ocr = Qwen2VLPlateOcr(
    model_name="D:/ComputerVisionNew/experiments/qwen2vl_crops_lora",
    device="cpu",
    use_lora_adapter=False,
)

# Test on sample image
img_path = "data/samples/sample_1_30G12345.jpg"
print(f"\nTest on: {img_path}")

img = Image.open(img_path).convert('RGB')
frame = np.array(img)

# Detect
frame_data = FrameData(image_id="test", frame=frame, source=img_path)
dets = detector.predict(frame_data)

print(f"\nDetections: {len(dets)}")
for d in dets:
    print(f"  Score: {d.score:.3f}, BBox: {d.bbox_xyxy}")

if dets:
    best = max(dets, key=lambda x: x.score)
    print(f"\nBest detection: score={best.score:.3f}")
    
    # Crop and preprocess
    plate_crop = crop_plate(frame_data, best, margin_ratio=0.05)
    print(f"Crop size: {plate_crop.crop.shape}")
    
    # Show crop
    Image.fromarray(plate_crop.crop).save("debug_crop.png")
    print("Saved debug_crop.png")
    
    # OCR
    print("\nRunning OCR...")
    prepared = preprocess_plate(plate_crop.crop, use_clahe=False)
    ocr_out = ocr.recognize(plate_crop, prepared)
    
    print(f"\nOCR Raw: '{ocr_out.text_raw}'")
    print(f"OCR Norm: '{ocr_out.text_norm}'")
    
    plate_text = advanced_repair_ocr_text(ocr_out.text_norm or normalize_plate_text(ocr_out.text_raw))
    print(f"Final: '{plate_text}'")
