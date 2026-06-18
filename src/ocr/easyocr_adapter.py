"""EasyOCR adapter for Vietnamese plates."""
import time
import numpy as np
import easyocr
from src.utils.types import OcrResult, PlateCrop


class EasyOcrAdapter:
    """Fast CPU-based OCR using EasyOCR for Vietnamese license plates."""
    
    def __init__(self, lang_list: list[str] = None, use_gpu: bool = False):
        if lang_list is None:
            lang_list = ["en", "vi"]
        self.reader = easyocr.Reader(lang_list, gpu=use_gpu, verbose=False)
    
    def recognize(self, plate_crop: PlateCrop, preprocessed: np.ndarray = None) -> OcrResult:
        """Run OCR on plate crop."""
        img = preprocessed if preprocessed is not None else plate_crop.crop
        
        # Run EasyOCR
        results = self.reader.readtext(img)
        
        # Extract text
        if results:
            # Combine all detected text
            texts = []
            for bbox, text, score in results:
                if text.strip():
                    texts.append((text.strip(), score))
            
            if texts:
                # Sort by position (left to right)
                full_text = "".join([t[0] for t in texts])
                avg_score = np.mean([t[1] for t in texts])
                
                return OcrResult(
                    image_id=plate_crop.image_id,
                    text_raw=full_text,
                    text_norm=full_text,
                    ocr_score=float(avg_score),
                )
        
        return OcrResult(
            image_id=plate_crop.image_id,
            text_raw="",
            text_norm="",
            ocr_score=0.0,
        )
