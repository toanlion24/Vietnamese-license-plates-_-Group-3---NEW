"""Sinh CSV nhãn mẫu (cột file + plate) để bạn điền GT rồi chạy ``build_test_manifest_from_folder.py``.

Mỗi dòng ``plate`` mặc định là ``TODO__<stem>`` — thay bằng biển đúng (vd. ``59G11234``) rồi chạy lại ghép manifest.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collect(images_dir: Path, *, recursive: bool) -> list[Path]:
    if not images_dir.is_dir():
        raise SystemExit(f"Không phải thư mục: {images_dir}")
    out: list[Path] = []
    if recursive:
        for p in sorted(images_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p.resolve())
    else:
        for p in sorted(images_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p.resolve())
    return out


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/manifests/img_gt_TEMPLATE.csv"),
        help="CSV: cột file (tên file), plate (GT — sửa sau).",
    )
    p.add_argument(
        "--file-column",
        choices=("basename", "relpath"),
        default="basename",
        help="basename: chỉ tên file (khớp --csv-key-format stem). relpath: đường dẫn tương đối images-dir.",
    )
    args = p.parse_args()

    root = args.images_dir.resolve()
    images = _collect(root, recursive=args.recursive)
    if not images:
        raise SystemExit("Không có ảnh.")

    out_path = args.out
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for ip in images:
        if args.file_column == "basename":
            file_cell = ip.name
        else:
            file_cell = str(ip.relative_to(root)).replace("\\", "/")
        rows.append((file_cell, f"TODO__{ip.stem}"))

    with out_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "plate"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_path}")
    print("Next: edit column plate (replace TODO__... with real plate text), then:")
    print(
        f'  python scripts/build_test_manifest_from_folder.py --images-dir "{root}" '
        f'--recursive --labels-csv "{out_path}" --output data/manifests/real_plates_manifest.csv'
    )


if __name__ == "__main__":
    main()
