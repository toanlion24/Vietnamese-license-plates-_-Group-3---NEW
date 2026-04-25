from __future__ import annotations

import argparse
from pathlib import Path
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image manifest for experiments.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("data/manifest.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows = [
        {"image_id": p.stem, "image_path": str(p)}
        for p in sorted(args.input_dir.glob("*"))
        if p.suffix.lower() in image_exts
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest written: {args.output_csv} ({len(rows)} rows)")

