"""Test EasyOCR speed."""
import sys
import time
sys.path.insert(0, "D:/ComputerVisionNew")

from PIL import Image
import numpy as np
from src.ocr.easyocr_adapter import EasyOcrAdapter
from src.utils.types import PlateCrop

print("Loading EasyOCR...")
t0 = time.time()
ocr = EasyOcrAdapter(lang_list=["en", "vi"])
print(f"Loaded: {time.time()-t0:.1f}s")

# Test on sample
img = Image.open("data/samples/sample_1_30G12345.jpg").convert('RGB')
img_array = np.array(img)

# Crop a region (simulating plate crop)
crop = img_array[195:286, 164:478]

# Save crop for visual check
Image.fromarray(crop).save("test_plate_crop.png")

# Run OCR
print("\nRunning OCR on plate crop...")
t0 = time.time()
result = ocr.recognize(PlateCrop("test", crop, (0,0,314,91), 0.8), crop)
print(f"OCR time: {(time.time()-t0)*1000:.0f}ms")
print(f"Result: '{result.text_raw}'")
