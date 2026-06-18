"""
Manual annotation tool cho plate images - Windows compatible.

Chạy trên terminal, dùng matplotlib để hiển thị ảnh.

Usage:
    python scripts/annotate_plates.py --input-dir data/images/raw --output data/labels_manual.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_existing_labels(csv_path: Path) -> dict[str, str]:
    """Load existing labels từ CSV."""
    labels = {}
    if csv_path and csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["image_id"]] = row["text_gt"]
        logger.info(f"Loaded {len(labels)} existing labels from {csv_path}")
    return labels


def save_labels(labels: dict[str, str], output_path: Path) -> None:
    """Save labels ra CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "text_gt"])
        writer.writeheader()
        for image_id, text in sorted(labels.items()):
            writer.writerow({"image_id": image_id, "text_gt": text})
    
    logger.info(f"Saved {len(labels)} labels to {output_path}")


def annotate_images(
    input_dir: Path,
    output_csv: Path,
    *,
    resume: bool = True,
    image_ext: str = ".png",
) -> dict[str, str]:
    """Annotate images với matplotlib display."""
    
    # Load existing labels if resume
    labels: dict[str, str] = {}
    if resume:
        labels = load_existing_labels(output_csv)
    
    # Find all images
    image_files = sorted(input_dir.glob(f"*{image_ext}"))
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    # Filter unannotated
    unannotated = [f for f in image_files if f.stem not in labels]
    logger.info(f"Already annotated: {len(labels)}")
    logger.info(f"Remaining to annotate: {len(unannotated)}")
    
    if not unannotated:
        logger.info("All images are annotated!")
        return labels
    
    # Disable interactive mode
    plt.ion()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.canvas.manager.set_window_title("Plate Annotation - Close this window to quit")
    
    print("\n" + "=" * 60)
    print("PLATE ANNOTATION TOOL")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Already done: {len(labels)}")
    print(f"Remaining: {len(unannotated)}")
    print("=" * 60)
    print("\nControls:")
    print("  Type plate text + [Enter]  - Save and go to next")
    print("  [Enter] with empty         - Mark as unreadable")
    print("  [n] + [Enter]              - Skip this image")
    print("  [q] + [Enter]              - Save and quit")
    print("  Close window               - Save and quit")
    print("=" * 60 + "\n")
    
    plt.show(block=False)
    
    try:
        for i, img_path in enumerate(unannotated):
            image_id = img_path.stem
            
            # Load and display image
            img = Image.open(img_path)
            
            ax.clear()
            ax.imshow(img)
            ax.set_title(f"{image_id} ({i+1}/{len(unannotated)})", fontsize=14)
            ax.axis("off")
            
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Prompt
            print(f"\n[{image_id}] ({i+1}/{len(unannotated)}) Enter plate text: ", end="", flush=True)
            
            # Read input
            try:
                text = input().strip().upper()
            except (EOFError, KeyboardInterrupt):
                break
            
            # Handle commands
            if text.lower() == "q":
                break
            elif text.lower() == "n":
                print("  Skipped")
                continue
            elif text == "":
                labels[image_id] = ""
                print("  Marked as unreadable")
            else:
                # Validate: remove invalid chars
                valid_chars = "0123456789ABCDEFGHIKLMNOPQRSTUVWXYZ"
                text = "".join(c for c in text if c in valid_chars)
                if text:
                    labels[image_id] = text
                    print(f"  Saved: {text}")
                else:
                    labels[image_id] = ""
                    print("  Marked as unreadable (no valid chars)")
            
            # Save periodically
            if len(labels) % 10 == 0:
                save_labels(labels, output_csv)
                print(f"  [Auto-saved {len(labels)} labels]")
    
    finally:
        plt.close("all")
        save_labels(labels, output_csv)
    
    return labels


def main():
    parser = argparse.ArgumentParser(description="Annotate plate images")
    parser.add_argument("--input-dir", type=Path, default=Path("data/images/raw"),
                        help="Input directory with images")
    parser.add_argument("--output", type=Path, default=Path("data/labels_manual.csv"),
                        help="Output CSV file")
    parser.add_argument("--ext", type=str, default=".png",
                        help="Image extension (default: .png)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing labels")
    
    args = parser.parse_args()
    
    annotate_images(
        args.input_dir,
        args.output,
        resume=not args.no_resume,
        image_ext=args.ext,
    )


if __name__ == "__main__":
    main()
