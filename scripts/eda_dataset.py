"""
Script 2: EDA - Exploratory Data Analysis cho dataset biển số VN
Usage:
    python scripts/eda_dataset.py --manifest data/manifests/manifest_fixed.csv --output reports/eda
"""

import argparse
import os
from pathlib import Path
from collections import Counter, defaultdict
import json
import math

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Image analysis will be skipped.")


def load_manifest(manifest_path: str) -> list[dict]:
    """Load manifest CSV."""
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = [h.strip() for h in lines[0].strip().split(',')]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split(',')
        if len(parts) >= len(header):
            row = dict(zip(header, parts[:len(header)]))
            data.append(row)
    
    return data


def analyze_image_sizes(manifest_data: list) -> dict:
    """Phân tích kích thước ảnh."""
    if not PIL_AVAILABLE:
        return {"error": "Pillow not available"}
    
    widths, heights = [], []
    aspect_ratios = []
    sizes = Counter()
    
    for item in manifest_data:
        img_path = item.get('image_path', '')
        if not img_path:
            continue
        
        full_path = Path(img_path)
        if not full_path.exists():
            full_path = Path('data') / img_path
        if not full_path.exists():
            continue
        
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
                size_key = f"{w}x{h}"
                sizes[size_key] += 1
        except Exception:
            continue
    
    if not widths:
        return {"error": "No images found"}
    
    return {
        "widths": {
            "min": min(widths),
            "max": max(widths),
            "mean": sum(widths) / len(widths),
            "median": sorted(widths)[len(widths) // 2]
        },
        "heights": {
            "min": min(heights),
            "max": max(heights),
            "mean": sum(heights) / len(heights),
            "median": sorted(heights)[len(heights) // 2]
        },
        "aspect_ratios": {
            "min": min(aspect_ratios),
            "max": max(aspect_ratios),
            "mean": sum(aspect_ratios) / len(aspect_ratios)
        },
        "unique_sizes": len(sizes),
        "most_common_sizes": dict(sizes.most_common(5)),
        "total_images": len(widths)
    }


def analyze_plate_texts(manifest_data: list) -> dict:
    """Phân tích text ground truth."""
    texts = [item.get('text_gt', '').strip() for item in manifest_data if item.get('text_gt')]
    
    lengths = [len(t) for t in texts]
    char_counts = Counter()
    digit_counts = Counter()
    letter_counts = Counter()
    
    for text in texts:
        for char in text:
            char_counts[char] += 1
            if char.isdigit():
                digit_counts[char] += 1
            elif char.isalpha():
                letter_counts[char.upper()] += 1
    
    plate_types = defaultdict(int)
    for text in texts:
        if '-' in text:
            plate_types['2-line'] += 1
        else:
            plate_types['1-line'] += 1
    
    unique_plates = len(set(texts))
    
    return {
        "total_samples": len(texts),
        "unique_plates": unique_plates,
        "lengths": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0
        },
        "most_common_length": Counter(lengths).most_common(1)[0] if lengths else (0, 0),
        "most_common_chars": dict(char_counts.most_common(20)),
        "most_common_digits": dict(digit_counts.most_common(10)),
        "most_common_letters": dict(letter_counts.most_common(10)),
        "plate_types": dict(plate_types),
        "sample_texts": texts[:20]
    }


def analyze_duplicate_plates(manifest_data: list) -> dict:
    """Phân tích các plate bị lặp lại."""
    plate_counts = Counter()
    for item in manifest_data:
        plate = item.get('text_gt', '').strip()
        if plate:
            plate_counts[plate] += 1
    
    duplicates = {p: c for p, c in plate_counts.items() if c > 1}
    
    return {
        "unique_plates": len(plate_counts),
        "plates_with_duplicates": len(duplicates),
        "total_samples": len(manifest_data),
        "max_occurrences": max(plate_counts.values()) if plate_counts else 0,
        "top_duplicates": dict(Counter(duplicates).most_common(20))
    }


def estimate_lightning_conditions(manifest_data: list) -> dict:
    """
    Ước tính điều kiện ánh sáng dựa trên tên file và metadata.
    (Đây là heuristic đơn giản - cần cải thiện với model thực tế)
    """
    conditions = Counter()
    
    for item in manifest_data:
        img_path = item.get('image_path', '').lower()
        
        if 'bright' in img_path or 'light' in img_path:
            conditions['bright'] += 1
        elif 'dark' in img_path or 'night' in img_path:
            conditions['dark'] += 1
        elif 'shadow' in img_path:
            conditions['shadow'] += 1
        elif 'blur' in img_path or 'motion' in img_path:
            conditions['motion_blur'] += 1
        elif 'rain' in img_path or 'wet' in img_path:
            conditions['rain/wet'] += 1
        else:
            conditions['normal'] += 1
    
    return {
        "distribution": dict(conditions),
        "total": sum(conditions.values()),
        "note": "This is estimated from filename patterns. For accurate analysis, use actual image analysis."
    }


def analyze_bbox_sizes(labels_dir: str) -> dict:
    """Phân tích kích thước bounding box từ YOLO labels."""
    if not os.path.exists(labels_dir):
        return {"error": f"Labels directory not found: {labels_dir}"}
    
    widths, heights = [], []
    areas = []
    img_width, img_height = 640, 640
    
    label_files = list(Path(labels_dir).glob("**/*.txt"))
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, _, _, w, h = [float(x) for x in parts[:5]]
                        
                        abs_w = w * img_width
                        abs_h = h * img_height
                        
                        widths.append(abs_w)
                        heights.append(abs_h)
                        areas.append(abs_w * abs_h)
        except Exception:
            continue
    
    if not widths:
        return {"error": "No valid labels found"}
    
    return {
        "bbox_widths": {
            "min": min(widths),
            "max": max(widths),
            "mean": sum(widths) / len(widths),
            "median": sorted(widths)[len(widths) // 2]
        },
        "bbox_heights": {
            "min": min(heights),
            "max": max(heights),
            "mean": sum(heights) / len(heights),
            "median": sorted(heights)[len(heights) // 2]
        },
        "bbox_areas": {
            "min": min(areas),
            "max": max(areas),
            "mean": sum(areas) / len(areas),
            "median": sorted(areas)[len(areas) // 2]
        },
        "total_boxes": len(widths),
        "total_images": len(label_files)
    }


def generate_html_report(eda_stats: dict, output_path: str):
    """Generate HTML report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VN License Plate Dataset - EDA Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; border-left: 4px solid #4CAF50; padding-left: 10px; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 3px solid #4CAF50; }
        .stat-label { color: #777; font-size: 12px; }
        .stat-value { color: #333; font-size: 24px; font-weight: bold; }
        .sample-list { background: #f9f9f9; padding: 10px; border-radius: 5px; font-family: monospace; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; }
        .chart-placeholder { background: linear-gradient(90deg, #e0e0e0, #f0f0f0); height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>VN License Plate Dataset - EDA Report</h1>
        <p>Generated automatically</p>
"""
    
    if 'image_sizes' in eda_stats and 'error' not in eda_stats.get('image_sizes', {}):
        html += """
        <h2>1. Image Size Analysis</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total Images</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Unique Sizes</div>
                <div class="stat-value">{unique_sizes}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Width</div>
                <div class="stat-value">{avg_width:.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Height</div>
                <div class="stat-value">{avg_height:.0f}</div>
            </div>
        </div>
""".format(
            total=eda_stats['image_sizes'].get('total_images', 0),
            unique_sizes=eda_stats['image_sizes'].get('unique_sizes', 0),
            avg_width=eda_stats['image_sizes']['widths']['mean'],
            avg_height=eda_stats['image_sizes']['heights']['mean']
        )
    
    if 'plate_texts' in eda_stats:
        pt = eda_stats['plate_texts']
        html += """
        <h2>2. Plate Text Analysis</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total Samples</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Unique Plates</div>
                <div class="stat-value">{unique}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Length</div>
                <div class="stat-value">{avg_len:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">1-Line / 2-Line</div>
                <div class="stat-value">{one_line} / {two_line}</div>
            </div>
        </div>
        <h3>Sample Plate Texts</h3>
        <div class="sample-list">{samples}</div>
""".format(
            total=pt['total_samples'],
            unique=pt['unique_plates'],
            avg_len=pt['lengths']['mean'],
            one_line=pt['plate_types'].get('1-line', 0),
            two_line=pt['plate_types'].get('2-line', 0),
            samples=', '.join(pt['sample_texts'][:30])
        )
    
    if 'bbox_stats' in eda_stats and 'error' not in eda_stats.get('bbox_stats', {}):
        bs = eda_stats['bbox_stats']
        html += """
        <h2>3. Bounding Box Analysis (YOLO Labels)</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Total Boxes</div>
                <div class="stat-value">{boxes}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Box Width</div>
                <div class="stat-value">{avg_w:.0f}px</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Box Height</div>
                <div class="stat-value">{avg_h:.0f}px</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Area</div>
                <div class="stat-value">{avg_area:.0f}px²</div>
            </div>
        </div>
""".format(
            boxes=bs['total_boxes'],
            avg_w=bs['bbox_widths']['mean'],
            avg_h=bs['bbox_heights']['mean'],
            avg_area=bs['bbox_areas']['mean']
        )
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  HTML report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EDA for VN License Plate Dataset")
    parser.add_argument('--manifest', type=str, default='data/manifests/manifest_fixed.csv',
                        help='Path to OCR manifest CSV')
    parser.add_argument('--labels-dir', type=str, default='data/labels/raw',
                        help='Path to YOLO labels directory')
    parser.add_argument('--crops-dir', type=str, default='data/crops',
                        help='Path to crops directory')
    parser.add_argument('--output', type=str, default='reports/eda',
                        help='Output directory for reports')
    parser.add_argument('--format', choices=['json', 'html', 'both'], default='both',
                        help='Output format')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EDA - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    os.makedirs(args.output, exist_ok=True)
    eda_stats = {}
    
    if os.path.exists(args.manifest):
        print(f"\n[1] Loading manifest: {args.manifest}")
        manifest_data = load_manifest(args.manifest)
        print(f"    Loaded {len(manifest_data)} samples")
        
        print("\n[2] Analyzing image sizes...")
        eda_stats['image_sizes'] = analyze_image_sizes(manifest_data)
        if 'error' not in eda_stats['image_sizes']:
            print(f"    Analyzed {eda_stats['image_sizes']['total_images']} images")
        
        print("\n[3] Analyzing plate texts...")
        eda_stats['plate_texts'] = analyze_plate_texts(manifest_data)
        print(f"    {eda_stats['plate_texts']['unique_plates']} unique plate texts")
        
        print("\n[4] Analyzing duplicate plates...")
        eda_stats['duplicates'] = analyze_duplicate_plates(manifest_data)
        print(f"    {eda_stats['duplicates']['plates_with_duplicates']} plates with duplicates")
        
        print("\n[5] Estimating lightning conditions...")
        eda_stats['lightning'] = estimate_lightning_conditions(manifest_data)
        print(f"    Estimated from filename patterns")
    
    if os.path.exists(args.labels_dir):
        print(f"\n[6] Analyzing YOLO bounding boxes...")
        eda_stats['bbox_stats'] = analyze_bbox_sizes(args.labels_dir)
        if 'error' not in eda_stats.get('bbox_stats', {}):
            print(f"    Analyzed {eda_stats['bbox_stats']['total_boxes']} bounding boxes")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if 'image_sizes' in eda_stats and 'error' not in eda_stats.get('image_sizes', {}):
        is_ = eda_stats['image_sizes']
        print(f"\nImage Sizes:")
        print(f"  Total: {is_['total_images']}")
        print(f"  Width: {is_['widths']['min']:.0f} - {is_['widths']['max']:.0f} (avg: {is_['widths']['mean']:.0f})")
        print(f"  Height: {is_['heights']['min']:.0f} - {is_['heights']['max']:.0f} (avg: {is_['heights']['mean']:.0f})")
    
    if 'plate_texts' in eda_stats:
        pt = eda_stats['plate_texts']
        print(f"\nPlate Texts:")
        print(f"  Total samples: {pt['total_samples']}")
        print(f"  Unique plates: {pt['unique_plates']}")
        print(f"  Avg length: {pt['lengths']['mean']:.1f} chars")
        print(f"  Types: 1-line={pt['plate_types'].get('1-line', 0)}, 2-line={pt['plate_types'].get('2-line', 0)}")
    
    if 'bbox_stats' in eda_stats and 'error' not in eda_stats.get('bbox_stats', {}):
        bs = eda_stats['bbox_stats']
        print(f"\nBounding Boxes:")
        print(f"  Total boxes: {bs['total_boxes']}")
        print(f"  Avg width: {bs['bbox_widths']['mean']:.0f}px")
        print(f"  Avg height: {bs['bbox_heights']['mean']:.0f}px")
        print(f"  Avg area: {bs['bbox_areas']['mean']:.0f}px²")
    
    if 'lightning' in eda_stats:
        lc = eda_stats['lightning']
        print(f"\nLightning Conditions (estimated):")
        for cond, count in lc['distribution'].items():
            print(f"  {cond}: {count}")
    
    output_json = os.path.join(args.output, 'eda_report.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(eda_stats, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON report: {output_json}")
    
    if args.format in ['html', 'both']:
        output_html = os.path.join(args.output, 'eda_report.html')
        generate_html_report(eda_stats, output_html)
    
    print("\n" + "=" * 60)
    print("Done! Reports saved to:", args.output)


if __name__ == "__main__":
    main()
