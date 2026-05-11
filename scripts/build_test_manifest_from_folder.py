"""Ghép manifest `data/test_manifest.csv` từ thư mục ảnh + nhãn (CSV, JSON hoặc file .txt).

Các cấu trúc thư mục thường gặp::

  (1) Một thư mục ảnh + một CSV chứa tên file và biển::

      mydata/images/img_01.jpg
      mydata/labels.csv
          file,plate
          img_01.jpg,51H12345

  (2) Cùng thư mục, mỗi ảnh có file nhãn cùng stem::

      mydata/img_01.jpg
      mydata/img_01.txt    # một dòng: 51H12345

  (3) Nhãn nằm trong thư mục con::

      mydata/images/img_01.jpg
      mydata/gt/img_01.txt

Đường dẫn trong manifest: **tương đối gốc project** nếu ảnh nằm trong project, ngược lại giữ **đường dẫn tuyệt đối**.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _norm_key_stem(s: str) -> str:
    return Path(s.strip()).stem.lower()


def _norm_key_name(s: str) -> str:
    return Path(s.strip()).name.lower()


def _collect_images(images_dir: Path, *, recursive: bool) -> list[Path]:
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


def _rel_or_abs(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        rel = path.relative_to(project_root.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_labels_csv(
    path: Path,
    *,
    key_column: str | None,
    gt_column: str | None,
    key_format: str,
) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise SystemExit("CSV nhãn không có header.")

        if gt_column:
            if gt_column not in reader.fieldnames:
                raise SystemExit(f"Không có cột GT: {gt_column}")
            gt_col = gt_column
        else:
            gt_col = ""
            for cand in ("gt", "text_gt", "label", "plate", "plate_text", "text", "transcript", "y"):
                if cand in reader.fieldnames:
                    gt_col = cand
                    break
            if not gt_col:
                raise SystemExit("Không đoán được cột GT. Dùng --gt-column.")

        if key_column:
            if key_column not in reader.fieldnames:
                raise SystemExit(f"Không có cột khóa: {key_column}")
            key_col = key_column
        else:
            key_col = ""
            for cand in (
                "filename",
                "file",
                "image",
                "image_path",
                "path",
                "file_path",
                "id",
                "image_id",
            ):
                if cand in reader.fieldnames:
                    key_col = cand
                    break
            if not key_col:
                raise SystemExit("Không đoán được cột file. Dùng --key-column.")

        by_key: dict[str, str] = {}
        for row in reader:
            raw_key = (row.get(key_col) or "").strip()
            gt = (row.get(gt_col) or "").strip()
            if not raw_key or not gt:
                continue
            if key_format == "stem":
                k = _norm_key_stem(raw_key)
            elif key_format == "name":
                k = _norm_key_name(raw_key)
            else:
                k = raw_key.replace("\\", "/").strip().lstrip("/").lower()
            by_key[k] = gt
        if not by_key:
            raise SystemExit("CSV nhãn không có dòng hợp lệ (khóa + gt).")
        return by_key


def _load_labels_json(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            fn = str(
                item.get("filename")
                or item.get("file")
                or item.get("image")
                or item.get("image_path")
                or item.get("id")
                or ""
            ).strip()
            gt = str(item.get("gt") or item.get("text_gt") or item.get("label") or "").strip()
            if fn and gt:
                out[_norm_key_stem(fn)] = gt
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str) and v.strip():
                out[_norm_key_stem(str(k))] = v.strip()
    else:
        raise SystemExit("JSON phải là list of objects hoặc dict.")
    if not out:
        raise SystemExit("JSON không có cặp filename/gt hợp lệ.")
    return out


def _read_sidecar_txt(txt_path: Path) -> str:
    if not txt_path.is_file():
        return ""
    text = txt_path.read_text(encoding="utf-8-sig").strip()
    return text.splitlines()[0].strip() if text else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", type=Path, required=True, help="Thư mục chứa ảnh.")
    p.add_argument("--recursive", action="store_true", help="Quét đệ quy thư mục con.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/test_manifest.csv"),
        help="Manifest đầu ra (mặc định: data/test_manifest.csv).",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Gốc project để sinh đường dẫn tương đối.",
    )
    p.add_argument(
        "--image-id",
        choices=("stem", "name"),
        default="stem",
        help="Trường image_id: stem hoặc tên file.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels-csv", type=Path, help="CSV: cột file + cột GT.")
    src.add_argument(
        "--labels-json",
        type=Path,
        help="JSON list [{filename, gt}] hoặc dict stem -> gt.",
    )
    src.add_argument(
        "--sidecar-dir",
        type=Path,
        help="Thư mục {stem}.txt (một dòng = biển).",
    )
    src.add_argument(
        "--sidecar-next-to-image",
        action="store_true",
        help="Đọc ảnh_stem.txt cùng thư mục với ảnh.",
    )

    p.add_argument("--key-column", type=str, default=None)
    p.add_argument("--gt-column", type=str, default=None)
    p.add_argument(
        "--csv-key-format",
        choices=("stem", "name", "relpath"),
        default="stem",
        help="Khớp khóa CSV với ảnh: theo stem, tên file, hoặc đường dẫn tương đối images-dir.",
    )
    p.add_argument("--strict", action="store_true", help="Lỗi nếu thiếu GT cho bất kỳ ảnh nào.")
    p.add_argument("--dry-run", action="store_true", help="Chỉ in thống kê, không ghi file.")
    return p.parse_args()


def _gt_for_image(
    img_path: Path,
    labels_map: dict[str, str],
    *,
    lookup_format: str,
    images_dir: Path,
) -> str:
    stem = _norm_key_stem(img_path.name)
    name = _norm_key_name(img_path.name)
    if lookup_format == "stem":
        return labels_map.get(stem, "")
    if lookup_format == "name":
        return labels_map.get(name, "")
    try:
        rel = img_path.resolve().relative_to(images_dir.resolve())
        rel_key = str(rel).replace("\\", "/").strip().lower()
        if rel_key in labels_map:
            return labels_map[rel_key]
        rel_key2 = rel_key.lstrip("./")
        if rel_key2 in labels_map:
            return labels_map[rel_key2]
    except ValueError:
        pass
    return labels_map.get(stem, "")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    project_root = args.project_root.resolve()
    images_dir = args.images_dir.resolve()

    images = _collect_images(images_dir, recursive=args.recursive)
    if not images:
        raise SystemExit(f"Không tìm thấy ảnh trong: {images_dir}")

    labels_map: dict[str, str] = {}
    lookup_format = "stem"

    if args.labels_csv:
        labels_map = _load_labels_csv(
            args.labels_csv.resolve(),
            key_column=args.key_column,
            gt_column=args.gt_column,
            key_format=args.csv_key_format,
        )
        if args.csv_key_format == "relpath":
            lookup_format = "relpath"
        elif args.csv_key_format == "name":
            lookup_format = "name"
        else:
            lookup_format = "stem"
    elif args.labels_json:
        labels_map = _load_labels_json(args.labels_json.resolve())
        lookup_format = "stem"
    elif args.sidecar_dir:
        sd = args.sidecar_dir.resolve()
        if not sd.is_dir():
            raise SystemExit(f"--sidecar-dir không phải thư mục: {sd}")
        for p in sorted(sd.glob("*.txt")):
            gt = _read_sidecar_txt(p)
            if gt:
                labels_map[_norm_key_stem(p.name)] = gt
        lookup_format = "stem"
    else:
        for img in images:
            gt = _read_sidecar_txt(img.with_suffix(".txt"))
            if gt:
                labels_map[_norm_key_stem(img.name)] = gt
        lookup_format = "stem"

    if not labels_map:
        raise SystemExit("Không có nhãn — kiểm tra CSV/JSON/sidecar.")

    rows: list[dict[str, str]] = []
    missing: list[str] = []

    for img in images:
        gt = _gt_for_image(img, labels_map, lookup_format=lookup_format, images_dir=images_dir)
        if not gt:
            missing.append(str(img))
            continue
        image_id = img.stem if args.image_id == "stem" else img.name
        rows.append(
            {
                "image_id": image_id,
                "image_path": _rel_or_abs(img, project_root),
                "gt": gt,
                "ambiguous_gt": "false",
            }
        )

    if args.strict and missing:
        raise SystemExit(
            "Thiếu GT cho ảnh:\n"
            + "\n".join(missing[:25])
            + (f"\n... tổng {len(missing)}" if len(missing) > 25 else "")
        )

    out_path = (project_root / args.output).resolve() if not args.output.is_absolute() else args.output
    print(f"Ảnh tìm thấy: {len(images)}")
    print(f"Dòng manifest (có GT): {len(rows)}")
    if missing:
        print(f"Cảnh báo: {len(missing)} ảnh không khớp nhãn (bỏ qua). Dùng --strict để lỗi.")
    if args.dry_run:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_id", "image_path", "gt", "ambiguous_gt"])
        w.writeheader()
        w.writerows(rows)
    print(f"Đã ghi: {out_path}")


if __name__ == "__main__":
    main()
