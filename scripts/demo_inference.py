"""
Simple CLI Demo for VN License Plate Recognition
==============================================
Run quick inference tests on images.

Usage:
    python scripts/demo_inference.py --input path/to/image.jpg
    python scripts/demo_inference.py --input path/to/images/ --batch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.yolov8_detector import YoloV8PlateDetector
from src.ocr.qwen_adapter import Qwen2VLPlateOcr
from src.pipeline.infer_plate_pipeline import PlateInferencePipeline
from src.utils.types import FrameData


def draw_results(image: np.ndarray, bbox: tuple, plate_text: str, confidence: float) -> np.ndarray:
    """Draw detection results on image."""
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Draw text with background
    text = f"{plate_text} ({confidence:.2f})"
    font_scale = 0.8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    
    cv2.rectangle(image, (x1, y1 - th - 15), (x1 + tw + 10, y1), (0, 255, 0), -1)
    cv2.putText(image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    
    return image


def process_image(
    pipeline: PlateInferencePipeline,
    image_path: str | Path,
    output_path: str | Path | None = None,
    show: bool = True
) -> dict:
    """Process single image and return results."""
    image_path = Path(image_path)
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Cannot read image {image_path}")
        return {}
    
    image_id = image_path.stem
    
    # Run inference
    print(f"🔄 Processing: {image_path.name}")
    start = time.time()
    
    frame_data = FrameData(
        image_id=image_id,
        frame=img,
        source=str(image_path),
    )
    
    result = pipeline.run(frame_data)
    elapsed = (time.time() - start) * 1000
    
    # Print results
    print(f"\n{'='*50}")
    print(f"📋 Results for: {image_path.name}")
    print(f"{'='*50}")
    
    if result.plate_text:
        print(f"✅ Plate Number: {result.plate_text}")
        print(f"📊 Confidence:  {result.confidence:.2%}")
        print(f"⏱️  Latency:    {result.timestamp_ms:.0f}ms")
        print(f"📐 BBox:        {result.bbox_xyxy}")
    else:
        print(f"⚠️  No plate detected")
    
    print(f"{'='*50}\n")
    
    # Draw and save results
    if result.bbox_xyxy:
        annotated = draw_results(img.copy(), result.bbox_xyxy, result.plate_text or "?", result.confidence)
    else:
        annotated = img.copy()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"💾 Saved: {output_path}")
    
    if show:
        # Save temp and show with PIL
        temp_path = Path("/tmp/plate_result.jpg")
        cv2.imwrite(str(temp_path), annotated)
        Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).show()
    
    return {
        "image_id": image_id,
        "plate_text": result.plate_text,
        "confidence": result.confidence,
        "bbox": result.bbox_xyxy,
        "latency_ms": result.timestamp_ms,
        "timestamp_ms": elapsed,
    }


def process_batch(
    pipeline: PlateInferencePipeline,
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.jpg",
    max_images: int | None = None
) -> list[dict]:
    """Process batch of images."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = sorted(input_dir.glob(pattern))
    if not image_files:
        image_files = sorted(input_dir.glob("*.png"))
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return []
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"📁 Found {len(image_files)} images")
    print(f"📤 Output: {output_dir}")
    print()
    
    results = []
    total_time = 0
    
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] ", end="", flush=True)
        
        start = time.time()
        result = process_image(
            pipeline,
            img_path,
            output_dir / f"{img_path.stem}_result.jpg",
            show=False
        )
        total_time += time.time() - start
        
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Batch Summary")
    print(f"{'='*50}")
    print(f"Total images:    {len(image_files)}")
    print(f"Processed:       {len(results)}")
    print(f"Detected:        {len([r for r in results if r.get('plate_text')])}")
    print(f"Total time:      {total_time:.2f}s")
    print(f"Avg per image:   {(total_time/len(results))*1000:.0f}ms")
    print(f"{'='*50}")
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Results saved: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VN License Plate Recognition Demo")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=str, help="Single image path")
    input_group.add_argument("--input-dir", "-d", type=str, help="Directory of images")
    
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path or directory")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch mode (when using --input-dir)")
    parser.add_argument("--max-images", "-n", type=int, default=None, help="Max images to process")
    
    # Model paths
    parser.add_argument("--yolo-model", type=str, 
                       default=str(PROJECT_ROOT / "runs/detect/experiments/detector/yolov8n_augmented/weights/best.pt"))
    parser.add_argument("--lora-path", type=str,
                       default=str(PROJECT_ROOT / "experiments/qwen2vl_crops_lora"))
    parser.add_argument("--use-lora", action="store_true", default=True, help="Use LoRA adapter")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA adapter")
    
    # Detection settings
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--margin", type=float, default=0.05, help="Crop margin ratio")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        if args.input:
            p = Path(args.input)
            args.output = str(p.parent / f"{p.stem}_result.jpg")
        else:
            args.output = str(PROJECT_ROOT / "outputs/demo")
    
    print("="*50)
    print("🚗 VN License Plate Recognition Demo")
    print("="*50)
    
    # Load models
    print("\n📦 Loading models...")
    
    detector = YoloV8PlateDetector(
        model_path=args.yolo_model,
        conf_threshold=args.conf,
    )
    print(f"  ✅ Detector: {args.yolo_model}")
    
    use_lora = args.use_lora and not args.no_lora
    ocr = Qwen2VLPlateOcr(
        model_name=args.lora_path if use_lora else "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        device="cuda",
        use_lora_adapter=use_lora,
    )
    print(f"  ✅ OCR: {'Qwen2-VL + LoRA' if use_lora else 'Qwen2-VL Base'}")
    
    pipeline = PlateInferencePipeline(
        detector=detector,
        ocr=ocr,
        crop_margin_ratio=args.margin,
    )
    
    print("\n🚀 Starting inference...\n")
    
    # Run
    if args.input:
        result = process_image(
            pipeline,
            args.input,
            args.output if not args.batch else None,
            show=False
        )
    else:
        results = process_batch(
            pipeline,
            args.input_dir,
            args.output,
            max_images=args.max_images
        )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
