from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal clean + simple balancing for VN license plate dataset."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/eda/clean_balance_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_path_candidates_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> list[Path]:
    relative_image = image_path.relative_to(images_dir)
    rel_txt = relative_image.with_suffix(".txt")
    candidates: list[Path] = [labels_dir / rel_txt]
    if len(relative_image.parts) > 1:
        candidates.append(labels_dir / Path(*relative_image.parts[1:]).with_suffix(".txt"))
    candidates.append(labels_dir / f"{image_path.stem}.txt")
    return candidates


def resolve_label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    candidates = label_path_candidates_for_image(image_path, images_dir, labels_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_and_validate_label_file(
    label_path: Path,
) -> tuple[list[tuple[int, float, float, float, float]], int]:
    boxes: list[tuple[int, float, float, float, float]] = []
    invalid_lines = 0
    if not label_path.exists():
        return boxes, invalid_lines
    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        return boxes, invalid_lines
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            invalid_lines += 1
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            invalid_lines += 1
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            invalid_lines += 1
            continue
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes, invalid_lines


def brightness_tag(gray_mean: float) -> str:
    if gray_mean < 70:
        return "dark"
    if gray_mean > 170:
        return "bright"
    return "normal"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed. Run: pip install -r requirements.txt") from exc

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")

    images = list_images(args.images_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.images_dir}")

    project_root = Path.cwd().resolve()
    kept_rel_paths: list[str] = []
    bucket_to_rel_paths: dict[str, list[str]] = {"dark": [], "normal": [], "bright": []}
    dropped_missing_labels = 0
    dropped_bad_images = 0
    dropped_invalid_labels = 0
    invalid_label_lines = 0

    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            dropped_bad_images += 1
            continue

        label_path = resolve_label_path_for_image(image_path, args.images_dir, args.labels_dir)
        boxes, invalid_lines = load_and_validate_label_file(label_path)
        invalid_label_lines += invalid_lines
        if not label_path.exists():
            dropped_missing_labels += 1
            continue
        if not boxes:
            dropped_invalid_labels += 1
            continue

        rel_path = str(image_path.resolve().relative_to(project_root))
        kept_rel_paths.append(rel_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bucket_to_rel_paths[brightness_tag(float(gray.mean()))].append(rel_path)

    non_empty_bucket_sizes = [len(v) for v in bucket_to_rel_paths.values() if v]
    target_per_bucket = min(non_empty_bucket_sizes) if non_empty_bucket_sizes else 0
    rng = random.Random(args.seed)

    balanced_rel_paths: list[str] = []
    for bucket_name in ("dark", "normal", "bright"):
        bucket_items = list(bucket_to_rel_paths[bucket_name])
        if not bucket_items:
            continue
        rng.shuffle(bucket_items)
        balanced_rel_paths.extend(bucket_items[:target_per_bucket])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_manifest = args.output_dir / "clean_images.txt"
    balanced_manifest = args.output_dir / "balanced_images.txt"
    write_lines(clean_manifest, kept_rel_paths)
    write_lines(balanced_manifest, balanced_rel_paths)

    report = {
        "images_total": len(images),
        "images_kept": len(kept_rel_paths),
        "dropped": {
            "missing_labels": dropped_missing_labels,
            "bad_images": dropped_bad_images,
            "invalid_or_empty_labels": dropped_invalid_labels,
            "invalid_label_lines": invalid_label_lines,
        },
        "brightness_buckets_before_balance": {
            k: len(v) for k, v in bucket_to_rel_paths.items()
        },
        "target_per_bucket_after_balance": target_per_bucket,
        "balanced_images_total": len(balanced_rel_paths),
        "outputs": {
            "clean_manifest": str(clean_manifest),
            "balanced_manifest": str(balanced_manifest),
        },
        "seed": args.seed,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Clean + balance completed")
    print(f"images_total={len(images)} kept={len(kept_rel_paths)}")
    print(f"balanced_images_total={len(balanced_rel_paths)}")
    print(f"clean_manifest={clean_manifest}")
    print(f"balanced_manifest={balanced_manifest}")
    print(f"report_path={args.report_path}")


if __name__ == "__main__":
    main()
