from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test split text files from image folder."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output folder for train.txt/val.txt/test.txt.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute file paths instead of relative paths.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Ratios must be non-negative.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}.")


def to_output_path(image_path: Path, root: Path, absolute: bool) -> str:
    return str(image_path.resolve()) if absolute else str(image_path.resolve().relative_to(root.resolve()))


def write_split(file_path: Path, items: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(items)
    file_path.write_text(content + ("\n" if items else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    images = sorted(
        p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_images = images[:n_train]
    val_images = images[n_train : n_train + n_val]
    test_images = images[n_train + n_val :]

    project_root = Path.cwd()
    train_items = [to_output_path(p, project_root, args.absolute_paths) for p in train_images]
    val_items = [to_output_path(p, project_root, args.absolute_paths) for p in val_images]
    test_items = [to_output_path(p, project_root, args.absolute_paths) for p in test_images]

    write_split(args.output_dir / "train.txt", train_items)
    write_split(args.output_dir / "val.txt", val_items)
    write_split(args.output_dir / "test.txt", test_items)

    print("Split completed")
    print(f"total={n}")
    print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"output_dir={args.output_dir}")
    print(f"seed={args.seed}")
    print(f"ratios=train:{args.train_ratio}, val:{args.val_ratio}, test:{args.test_ratio}")

    if n_test != 0 and len(test_items) == 0:
        print("warning=test split ended up empty due to low sample size")


if __name__ == "__main__":
    main()
