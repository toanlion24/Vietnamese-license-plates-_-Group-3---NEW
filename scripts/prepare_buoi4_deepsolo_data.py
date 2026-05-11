from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.plate_rules import normalize_plate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Buoi 4 manifest rows into simple DeepSolo-style text spotting annotations."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/manifests/buoi4_deepsolo.example.csv"),
        help="CSV with image_id, image_path, text_gt and bbox_xyxy or polygon.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/deepsolo/buoi4"),
        help="Output folder for DeepSolo annotation txt/jsonl files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name to keep from manifest. Use 'all' to export every row.",
    )
    return parser.parse_args()


def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def _parse_number_list(raw: str, expected_lengths: tuple[int, ...]) -> list[float]:
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(values) not in expected_lengths:
        expected = " hoặc ".join(str(length) for length in expected_lengths)
        raise ValueError(f"Expected {expected} numeric values, got {len(values)} from: {raw}")
    return [float(value) for value in values]


def _bbox_to_polygon(raw_bbox: str) -> list[float]:
    x1, y1, x2, y2 = _parse_number_list(raw_bbox, (4,))
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _row_to_polygon(row: dict[str, str]) -> list[float]:
    polygon = _first_non_empty(row, ("polygon", "points", "poly"))
    if polygon:
        return _parse_number_list(polygon, (8,))

    bbox = _first_non_empty(row, ("bbox_xyxy", "bbox", "box"))
    if bbox:
        return _bbox_to_polygon(bbox)

    raise ValueError("Each row must contain either polygon or bbox_xyxy.")


def _format_point(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def convert_manifest(manifest_csv: Path, output_dir: Path, split: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_txt = output_dir / f"{split}_annotations.txt"
    annotation_jsonl = output_dir / f"{split}_annotations.jsonl"

    exported = 0
    skipped = 0
    errors: list[str] = []

    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as fp, annotation_txt.open(
        "w", encoding="utf-8", newline="\n"
    ) as txt_fp, annotation_jsonl.open("w", encoding="utf-8", newline="\n") as jsonl_fp:
        reader = csv.DictReader(fp)
        for row_idx, row in enumerate(reader, start=2):
            row_split = row.get("split", "").strip()
            if split != "all" and row_split and row_split != split:
                skipped += 1
                continue

            try:
                image_id = _first_non_empty(row, ("image_id", "id", "filename", "file_name"))
                image_path = _first_non_empty(row, ("image_path", "path", "file_path"))
                text_gt = _first_non_empty(row, ("text_gt", "gt", "label", "transcript"))
                polygon = _row_to_polygon(row)
                text_norm = normalize_plate_text(text_gt)
                if not image_id:
                    image_id = Path(image_path).stem
                if not image_path or not text_norm:
                    raise ValueError("Missing image_path or text_gt.")

                polygon_text = ",".join(_format_point(value) for value in polygon)
                txt_fp.write(f"{image_path}\t{polygon_text},{text_norm}\n")
                jsonl_fp.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "polygon": polygon,
                            "text": text_norm,
                            "split": row_split or split,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                exported += 1
            except ValueError as exc:
                skipped += 1
                errors.append(f"row {row_idx}: {exc}")

    summary = {
        "manifest_csv": str(manifest_csv),
        "output_dir": str(output_dir),
        "split": split,
        "exported": exported,
        "skipped": skipped,
        "annotation_txt": str(annotation_txt),
        "annotation_jsonl": str(annotation_jsonl),
        "errors": errors,
    }
    (output_dir / f"{split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = convert_manifest(args.manifest_csv, args.output_dir, args.split)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
