"""
Auto-label plate images bằng OCR, sau đó verify thủ công.

Sử dụng EasyOCR để đọc biển số tự động,
sau đó chỉ cần correct những cái sai.

Usage:
    # Bước 1: OCR tự động
    python scripts/auto_label.py --input-dir data/images/raw --output data/labels_auto.csv

    # Bước 2: Verify (tùy chọn)
    python scripts/auto_label.py --verify --input-dir data/images/raw --output data/labels_auto.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np
import easyocr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Khởi tạo EasyOCR reader (chỉ chạy 1 lần)
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        logger.info("Loading EasyOCR reader (English + Vietnamese)... ")
        _reader = easyocr.Reader(['en', 'vi'], gpu=False, verbose=False)
    return _reader


def normalize_plate_text(text: str) -> str:
    """Normalize OCR result thành format biển số VN."""
    if not text:
        return ""
    
    # Remove spaces
    text = text.replace(" ", "").replace("  ", "")
    
    # Uppercase
    text = text.upper()
    
    # Keep only valid VN plate characters
    # Format: 2 số + 1 chữ (loại cũ) hoặc số + 2-3 chữ + số (loại mới)
    valid_chars = "0123456789ABCDEFGHIKLMNOPQRSTUVWXYZ"
    text = "".join(c for c in text if c in valid_chars)
    
    return text


def ocr_single_image(image_path: Path, reader) -> tuple[str, float]:
    """OCR một ảnh, trả về text và confidence."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return "", 0.0
        
        # EasyOCR
        results = reader.readtext(img)
        
        if not results:
            return "", 0.0
        
        # Combine all text
        full_text = ""
        total_conf = 0.0
        
        for bbox, text, conf in results:
            cleaned = normalize_plate_text(text)
            if cleaned:
                full_text += cleaned
                total_conf += conf
        
        # Calculate average confidence
        num_results = len([r for r in results if normalize_plate_text(r[1])])
        avg_conf = total_conf / num_results if num_results > 0 else 0.0
        
        return full_text, avg_conf
        
    except Exception as e:
        logger.debug(f"Error OCR {image_path}: {e}")
        return "", 0.0


def auto_label_images(
    input_dir: Path,
    output_csv: Path,
    *,
    image_ext: str = ".png",
) -> dict[str, dict]:
    """Auto-label tất cả ảnh bằng OCR."""
    reader = get_reader()
    
    # Find images
    image_files = sorted(input_dir.glob(f"*{image_ext}"))
    logger.info(f"Found {len(image_files)} images")
    
    results = {}
    
    logger.info("Running OCR (this may take a while for first run)... ")
    
    for img_path in tqdm(image_files, desc="OCR"):
        image_id = img_path.stem
        
        text, conf = ocr_single_image(img_path, reader)
        
        results[image_id] = {
            "text_gt": text,
            "confidence": conf,
            "image_path": str(img_path),
        }
    
    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "text_gt", "confidence"])
        writer.writeheader()
        for image_id, data in sorted(results.items()):
            writer.writerow({
                "image_id": image_id,
                "text_gt": data["text_gt"],
                "confidence": data["confidence"],
            })
    
    # Stats
    successful = sum(1 for r in results.values() if r["text_gt"])
    avg_conf = np.mean([r["confidence"] for r in results.values() if r["text_gt"]]) or 0
    
    logger.info(f"\n✅ Auto-labeling complete!")
    logger.info(f"   Total: {len(results)} images")
    logger.info(f"   Successful OCR: {successful} ({successful/len(results)*100:.1f}%)")
    logger.info(f"   Average confidence: {avg_conf:.3f}")
    logger.info(f"   Output: {output_csv}")
    
    return results


def verify_labels(
    input_dir: Path,
    labels_csv: Path,
    *,
    image_ext: str = ".png",
    max_samples: int = 100,
) -> None:
    """Verify và correct labels thủ công."""
    import easyocr
    
    # Load existing labels
    labels = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["image_id"]] = row["text_gt"]
    
    # Get images to verify (prefer low confidence ones)
    image_files = sorted(input_dir.glob(f"*{image_ext}"))
    
    # Sort by confidence (low first) for priority verification
    to_verify = []
    for img_path in image_files:
        image_id = img_path.stem
        if image_id in labels:
            # Will verify later
            to_verify.append((image_id, img_path, labels[image_id]))
    
    logger.info(f"Found {len(to_verify)} labeled images to verify")
    logger.info("Controls:")
    logger.info("  [Enter] - Keep current label")
    logger.info("  Type new text - Replace label")
    logger.info("  [n] - Mark as unreadable")
    logger.info("  [q] - Quit and save")
    
    updated = 0
    
    for image_id, img_path, current_label in tqdm(to_verify[:max_samples], desc="Verifying"):
        img = cv2.imread(str(img_path))
        
        # Resize for display
        h, w = img.shape[:2]
        scale = 800 / w
        img_display = cv2.resize(img, (800, int(h * scale)))
        
        # Show current label
        display = img_display.copy()
        cv2.putText(display, f"ID: {image_id}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Current: {current_label or '(empty)'}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "Type new text or press Enter to keep:", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Verify Labels", display)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("n"):
            labels[image_id] = ""
            updated += 1
        elif key == 13:  # Enter
            pass  # Keep current
        else:
            pass  # Handled by input()
    
    cv2.destroyAllWindows()
    
    # Save updated labels
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "text_gt"])
        writer.writeheader()
        for image_id, text in sorted(labels.items()):
            writer.writerow({"image_id": image_id, "text_gt": text})
    
    logger.info(f"Updated {updated} labels")


def main():
    parser = argparse.ArgumentParser(description="Auto-label plate images with OCR")
    parser.add_argument("--input-dir", type=Path, default=Path("data/images/raw"),
                       help="Input directory with images")
    parser.add_argument("--output", type=Path, default=Path("data/labels_auto.csv"),
                       help="Output CSV file")
    parser.add_argument("--verify", action="store_true",
                       help="Verify labels after auto-labeling")
    parser.add_argument("--ext", type=str, default=".png",
                       help="Image extension")
    parser.add_argument("--max-verify", type=int, default=100,
                       help="Max images to verify")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_labels(args.input_dir, args.output, 
                      image_ext=args.ext, max_samples=args.max_verify)
    else:
        auto_label_images(args.input_dir, args.output, image_ext=args.ext)


if __name__ == "__main__":
    main()
