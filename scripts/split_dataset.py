"""
Script 1: Split Data - Tách train/val/test cho manifests và YOLO labels
Usage:
    python scripts/split_dataset.py --manifest data/manifests/manifest_fixed.csv --output data/splits
"""

import argparse
import random
import os
from pathlib import Path
from collections import defaultdict


def load_manifest(manifest_path: str) -> list[dict]:
    """Load manifest CSV và trả về list of dicts."""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split(',')
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split(',')
        row = dict(zip(header, parts))
        data.append(row)
    
    return data


def split_by_ratio(data: list, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                   seed: int = 42) -> tuple[list, list, list]:
    """Split data theo tỉ lệ train/val/test."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    return train, val, test


def split_by_unique_plates(data: list, train_ratio: float = 0.8, val_ratio: float = 0.1,
                           seed: int = 42) -> tuple[list, list, list]:
    """Split data giữ nguyên group (cùng plate text) trong cùng split."""
    random.seed(seed)
    
    groups = defaultdict(list)
    for item in data:
        plate_text = item.get('text_gt', item.get('plate_text', ''))
        groups[plate_text].append(item)
    
    unique_plates = list(groups.keys())
    random.shuffle(unique_plates)
    
    n = len(unique_plates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_plates = unique_plates[:n_train]
    val_plates = unique_plates[n_train:n_train + n_val]
    test_plates = unique_plates[n_train + n_val:]
    
    train = [item for p in train_plates for item in groups[p]]
    val = [item for p in val_plates for item in groups[p]]
    test = [item for p in test_plates for item in groups[p]]
    
    return train, val, test


def save_split_files(split_data: dict, output_dir: str, manifest_name: str = ""):
    """Lưu train/val/test splits ra file txt."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for split_name, items in split_data.items():
        filepath = output_path / f"{split_name}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in items:
                image_path = item.get('image_path', '')
                if image_path:
                    f.write(f"{image_path}\n")
        
        results[split_name] = len(items)
        print(f"  {split_name}: {len(items)} items -> {filepath}")
    
    return results


def save_manifest_with_split(manifest_data: list, split_assignments: dict, output_path: str):
    """Lưu manifest CSV với cột split mới."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("image_id,image_path,text_gt,split\n")
        for item in manifest_data:
            image_id = item.get('image_id', '')
            image_path = item.get('image_path', '')
            text_gt = item.get('text_gt', '')
            split = split_assignments.get(image_id, 'unknown')
            f.write(f"{image_id},{image_path},{text_gt},{split}\n")
    
    print(f"  Updated manifest: {output_path}")


def generate_yolo_splits(yolo_root: str, train_ratio: float = 0.8, val_ratio: float = 0.1,
                         seed: int = 42) -> dict:
    """Generate train/val/test splits cho YOLO dataset."""
    yolo_path = Path(yolo_root)
    images_dir = yolo_path / "images"
    labels_dir = yolo_path / "labels"
    
    if not images_dir.exists():
        print(f"  YOLO images dir not found: {images_dir}")
        return {}
    
    image_files = list(images_dir.glob("**/*.jpg")) + list(images_dir.glob("**/*.png"))
    image_files = [f for f in image_files if f.is_file()]
    
    random.seed(seed)
    random.shuffle(image_files)
    
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }
    
    results = {}
    for split_name, files in splits.items():
        output_file = yolo_path / f"{split_name}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for img_file in files:
                rel_path = str(img_file.relative_to(yolo_path.parent))
                f.write(f"{rel_path}\n")
        
        results[split_name] = len(files)
        print(f"  YOLO {split_name}: {len(files)} images -> {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Split dataset for YOLO and OCR training")
    parser.add_argument('--manifest', type=str, default='data/manifests/manifest_fixed.csv',
                        help='Path to OCR manifest CSV')
    parser.add_argument('--yolo-root', type=str, default='data/yolo_dataset_full',
                        help='Path to YOLO dataset root')
    parser.add_argument('--output', type=str, default='data/splits',
                        help='Output directory for split files')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train split ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--split-by-plate', action='store_true',
                        help='Split by unique plate text (keeps same plates together)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATASET SPLIT SCRIPT")
    print("=" * 60)
    print(f"Train ratio: {args.train_ratio}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Test ratio: {1 - args.train_ratio - args.val_ratio:.2f}")
    print(f"Seed: {args.seed}")
    print()
    
    all_results = {}
    
    if os.path.exists(args.manifest):
        print(f"[1] Processing OCR Manifest: {args.manifest}")
        data = load_manifest(args.manifest)
        print(f"    Total items: {len(data)}")
        
        if args.split_by_plate:
            train, val, test = split_by_unique_plates(data, args.train_ratio, args.val_ratio, args.seed)
        else:
            train, val, test = split_by_ratio(data, args.train_ratio, args.val_ratio, args.seed)
        
        split_data = {'train': train, 'val': val, 'test': test}
        
        manifest_results = save_split_files(split_data, args.output, "manifest")
        
        split_assignments = {}
        for item in train:
            split_assignments[item.get('image_id', '')] = 'train'
        for item in val:
            split_assignments[item.get('image_id', '')] = 'val'
        for item in test:
            split_assignments[item.get('image_id', '')] = 'test'
        
        manifest_out = Path(args.output) / "manifest_with_split.csv"
        save_manifest_with_split(data, split_assignments, str(manifest_out))
        
        all_results['manifest'] = manifest_results
        print()
    
    if os.path.exists(args.yolo_root):
        print(f"[2] Processing YOLO Dataset: {args.yolo_root}")
        yolo_results = generate_yolo_splits(args.yolo_root, args.train_ratio, args.val_ratio, args.seed)
        all_results['yolo'] = yolo_results
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        total = 0
        for split, count in results.items():
            print(f"  {split}: {count}")
            total += count
        print(f"  TOTAL: {total}")
    
    print()
    print("Split files saved to:", args.output)
    print("Done!")


if __name__ == "__main__":
    main()
