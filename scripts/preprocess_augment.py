from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create basic preprocessed and augmented images for Buoi 2."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/images"))
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/labels/raw"),
        help="YOLO labels directory. If provided, transformed labels will be saved with augmented images.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/augmented"))
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-images", type=int, default=0, help="0 means process all images.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return cv2


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def save_image(path: Path, image, cv2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def load_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        boxes.append((cls_id, xc, yc, w, h))
    return boxes


def save_label_file(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cls_id, xc, yc, bw, bh in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def label_path_candidates_for_image(
    image_path: Path,
    images_dir: Path,
    labels_dir: Path,
) -> list[Path]:
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


def yolo_to_corners(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return x1, y1, x2, y2


def corners_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width - 1))
    y2 = max(0.0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh


def resize_boxes(
    boxes: list[tuple[int, float, float, float, float]],
) -> list[tuple[int, float, float, float, float]]:
    # YOLO boxes are normalized, resize keeps box values unchanged.
    return [tuple(box) for box in boxes]


def rotate_boxes(
    boxes: list[tuple[int, float, float, float, float]],
    width: int,
    height: int,
    matrix,
    np,
) -> list[tuple[int, float, float, float, float]]:
    rotated: list[tuple[int, float, float, float, float]] = []
    for cls_id, xc, yc, bw, bh in boxes:
        x1, y1, x2, y2 = yolo_to_corners(xc, yc, bw, bh, width, height)
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        transformed = (matrix @ corners_h.T).T
        tx1 = float(np.min(transformed[:, 0]))
        ty1 = float(np.min(transformed[:, 1]))
        tx2 = float(np.max(transformed[:, 0]))
        ty2 = float(np.max(transformed[:, 1]))
        rxc, ryc, rbw, rbh = corners_to_yolo(tx1, ty1, tx2, ty2, width, height)
        if rbw > 0 and rbh > 0:
            rotated.append((cls_id, rxc, ryc, rbw, rbh))
    return rotated


def apply_basic_transforms(
    image,
    boxes: list[tuple[int, float, float, float, float]],
    target_size: tuple[int, int],
    rng: np.random.Generator,
    cv2,
    np,
):
    width, height = target_size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    alpha = float(rng.uniform(0.85, 1.2))
    beta = int(rng.integers(-20, 21))
    brightness_contrast = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)

    angle = float(rng.uniform(-5.0, 5.0))
    h, w = brightness_contrast.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        brightness_contrast,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    resized_boxes = resize_boxes(boxes)
    brightness_boxes = resize_boxes(boxes)
    rotated_boxes = rotate_boxes(boxes, w, h, matrix, np)

    return {
        "resized": (resized, resized_boxes),
        "brightness_contrast": (brightness_contrast, brightness_boxes),
        "rotated": (rotated, rotated_boxes),
    }


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is not installed. Run: pip install -r requirements.txt"
        ) from exc
    cv2 = ensure_cv2()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.labels_dir is not None and not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")

    images = list_images(args.input_dir)
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")

    if args.max_images > 0:
        images = images[: args.max_images]

    rng = np.random.default_rng(args.seed)
    target_size = (args.size[0], args.size[1])

    total_saved = 0
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        boxes: list[tuple[int, float, float, float, float]] = []
        if args.labels_dir is not None:
            label_path = resolve_label_path_for_image(image_path, args.input_dir, args.labels_dir)
            boxes = load_label_file(label_path)

        transforms = apply_basic_transforms(img, boxes, target_size, rng, cv2, np)
        rel = image_path.relative_to(args.input_dir)
        stem = rel.with_suffix("")

        for name, (transformed, transformed_boxes) in transforms.items():
            out_path = args.output_dir / stem.parent / f"{stem.name}__{name}.jpg"
            save_image(out_path, transformed, cv2)
            if args.labels_dir is not None:
                label_out = out_path.with_suffix(".txt")
                save_label_file(label_out, transformed_boxes)
            total_saved += 1

    print("Preprocess/Augment completed")
    print(f"input_images={len(images)}")
    print(f"saved_images={total_saved}")
    print(f"output_dir={args.output_dir}")
    print(f"size={target_size}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
