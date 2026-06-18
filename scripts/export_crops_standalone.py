"""
Export plate crops cho training - Standalone script.

Không phụ thuộc vào cấu trúc src/

Usage:
    python scripts/export_crops_standalone.py
"""

import os
import csv
import zipfile
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_labels(csv_path: str) -> dict:
    """Load ground truth labels."""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image_id']
            labels[image_id] = row['text_gt']
    return labels


def export_crops(input_dir: str, output_dir: str, labels_csv: str, create_zip: bool = True):
    """Export plate crops."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels = load_labels(labels_csv)
    print(f"Loaded {len(labels)} labels")
    
    # Find images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    exported = 0
    skipped = 0
    
    for img_path in tqdm(image_files, desc="Exporting crops"):
        image_id = img_path.stem
        
        # Check if we have label
        if image_id not in labels:
            skipped += 1
            continue
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        
        # Crop: use full image as crop (already cropped plates)
        # Add small margin
        h, w = img.shape[:2]
        margin = int(min(h, w) * 0.05)
        crop = img[margin:h-margin, margin:w-margin]
        
        # Save crop
        output_path = output_dir / f"{image_id}.jpg"
        cv2.imwrite(str(output_path), crop)
        
        exported += 1
    
    print(f"\nExported: {exported}")
    print(f"Skipped: {skipped}")
    print(f"Output: {output_dir}")
    
    # Create manifest
    manifest_path = output_dir / "manifest.csv"
    manifest_images = list(output_dir.glob("*.jpg"))
    
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_path', 'text_gt'])
        for img_path in sorted(manifest_images):
            image_id = img_path.stem
            text_gt = labels.get(image_id, '')
            writer.writerow([image_id, str(img_path), text_gt])
    
    print(f"Manifest: {manifest_path}")
    
    # Create ZIP if requested
    if create_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        print(f"\nCreating ZIP: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_path in tqdm(manifest_images, desc="Zipping"):
                zipf.write(img_path, img_path.name)
            zipf.write(manifest_path, manifest_path.name)
        
        print(f"ZIP created: {zip_path}")
        print(f"ZIP size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    export_crops(
        input_dir="data/images/raw",
        output_dir="data/crops",
        labels_csv="data/labels_manual.csv",
        create_zip=True,
    )
