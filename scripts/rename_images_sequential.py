"""Đổi tên ảnh trong thư mục thành ``plate_0001.ext`` … cho dễ gán nhãn / manifest.

- Mặc định: **đệ quy** + **flatten** — gom mọi ảnh vào gốc ``--images-dir``, tên ``plate_0001`` … duy nhất.
- ``--no-flatten``: đổi tên trong từng thư mục (``plate_0001`` lặp lại theo thư mục).

Ghi ``data/manifests/rename_mapping.csv`` (``old_path``, ``new_path`` tương đối project).

Khi flatten, dùng thư mục tạm hai bước để tránh đè file khi ``src`` và ``dest`` cùng nằm dưới ``root``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collect(root: Path, *, recursive: bool) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    out: list[Path] = []
    if recursive:
        for p in sorted(root.rglob("*")):
            if "__rename_stage__" in p.parts:
                continue
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p.resolve())
    else:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p.resolve())
    return out


def _rel_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", type=Path, default=Path("data/img"))
    p.add_argument("--no-recursive", action="store_true", help="Chỉ tệp trực tiếp trong images-dir.")
    p.add_argument("--no-flatten", action="store_true", help="Không gom về gốc; đổi tên theo từng thư mục.")
    p.add_argument("--prefix", type=str, default="plate_", help="Tiền tố tên file.")
    p.add_argument(
        "--mapping-out",
        type=Path,
        default=Path("data/manifests/rename_mapping.csv"),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    recursive = not args.no_recursive
    flatten = not args.no_flatten

    root = (PROJECT_ROOT / args.images_dir).resolve() if not args.images_dir.is_absolute() else args.images_dir.resolve()
    images = sorted(_collect(root, recursive=recursive), key=lambda x: str(x).lower())
    if not images:
        raise SystemExit("No images found.")

    planned: list[tuple[Path, Path]] = []
    if flatten:
        for i, src in enumerate(images, start=1):
            ext = src.suffix.lower() or ".jpg"
            dest = root / f"{args.prefix}{i:04d}{ext}"
            if src.resolve() == dest.resolve():
                continue
            planned.append((src, dest))
    else:
        from collections import defaultdict

        by_parent: dict[Path, list[Path]] = defaultdict(list)
        for src in images:
            by_parent[src.parent].append(src)
        for parent in sorted(by_parent.keys(), key=lambda x: str(x).lower()):
            for i, src in enumerate(sorted(by_parent[parent], key=lambda x: x.name.lower()), start=1):
                ext = src.suffix.lower() or ".jpg"
                dest = parent / f"{args.prefix}{i:04d}{ext}"
                if src.resolve() == dest.resolve():
                    continue
                planned.append((src, dest))

    print(f"Found {len(images)} images, {len(planned)} renames (flatten={flatten}).")
    if args.dry_run:
        for src, dest in planned[:20]:
            print(f"  {_rel_project(src)} -> {_rel_project(dest)}")
        if len(planned) > 20:
            print(f"  ... and {len(planned) - 20} more")
        return

    if flatten:
        stage = root / "__rename_stage__"
        stage.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path, Path]] = []
        try:
            for i, (src, dest) in enumerate(planned):
                mid = stage / f"_s{i:05d}{src.suffix.lower() or '.jpg'}"
                shutil.move(str(src), str(mid))
                staged.append((src, mid, dest))
            for _src, mid, dest in staged:
                shutil.move(str(mid), str(dest))
        finally:
            try:
                if stage.is_dir() and not any(stage.iterdir()):
                    stage.rmdir()
            except OSError:
                pass
        mapping = [(_rel_project(s), _rel_project(d)) for s, d in planned]
    else:
        mapping = []
        for src, dest in planned:
            if dest.exists() and src.resolve() != dest.resolve():
                raise SystemExit(f"Target already exists: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            mapping.append((_rel_project(src), _rel_project(dest)))

    map_path = (PROJECT_ROOT / args.mapping_out).resolve() if not args.mapping_out.is_absolute() else args.mapping_out
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["old_path", "new_path"])
        w.writerows(mapping)
    print(f"Mapping written: {map_path}")

    if flatten:
        for dirpath, _dirnames, _filenames in os.walk(str(root), topdown=False):
            p = Path(dirpath)
            if p == root or p.name == "__rename_stage__":
                continue
            try:
                if p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
                    print(f"Removed empty dir: {p}")
            except OSError:
                pass

    print("Done. Regenerate GT template:")
    print(
        f'  python scripts/bootstrap_gt_csv_from_folder.py --images-dir "{root}" '
        f'{"--recursive" if recursive else ""} --out data/manifests/img_gt_TEMPLATE.csv'
    )


if __name__ == "__main__":
    main()
