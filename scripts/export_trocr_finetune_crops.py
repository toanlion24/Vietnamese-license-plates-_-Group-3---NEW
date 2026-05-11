"""Xuất ảnh crop biển + CSV cho ``scripts/train_trocr.py`` từ manifest có GT."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.base import DummyCenterDetector
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.preprocess.ops import crop_plate
from src.utils.types import FrameData


def _first_nonempty(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def load_manifest(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise SystemExit("Manifest không có header.")
        for idx, row in enumerate(reader, start=2):
            image_id = _first_nonempty(row, ("image_id", "id", "filename"))
            image_path = _first_nonempty(row, ("image_path", "path", "file_path"))
            gt = _first_nonempty(row, ("gt", "text_gt", "label", "plate_gt"))
            if not image_id or not image_path or not gt:
                raise SystemExit(f"Dòng {idx} thiếu image_id, image_path hoặc gt.")
            amb = str(row.get("ambiguous_gt", "")).strip().lower() in ("1", "true", "yes", "y")
            out.append({"image_id": image_id, "image_path": image_path, "gt": gt, "ambiguous_gt": amb})
    return out


def resolve_image_path(project_root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _rel_project(project_root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _build_detector(ns: argparse.Namespace):
    if ns.detector_backend == "dummy":
        return DummyCenterDetector()
    return YoloV8PlateDetector(
        model_path=ns.detector_model,
        conf_threshold=ns.detector_conf,
        iou=float(ns.yolo_iou),
        imgsz=int(ns.yolo_imgsz),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("data/trocr_finetune/crops"))
    p.add_argument("--out-csv", type=Path, default=Path("data/trocr_finetune/train_from_manifest.csv"))
    p.add_argument("--detector-backend", choices=["yolov8", "dummy"], default="yolov8")
    p.add_argument("--detector-model", type=Path, default=Path("weights/yolov8_license_plate.pt"))
    p.add_argument("--detector-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.45)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--crop-margin-ratio", type=float, default=0.08)
    p.add_argument("--skip-missing-detect", action="store_true", help="Bỏ qua ảnh không detect thay vì dừng lỗi.")
    args = p.parse_args()

    project_root = args.project_root.resolve()
    manifest = load_manifest(args.manifest.resolve() if not args.manifest.is_absolute() else args.manifest)
    out_dir = (project_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = (project_root / args.out_csv).resolve() if not args.out_csv.is_absolute() else args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    detector = _build_detector(args)
    csv_rows: list[dict[str, str]] = []
    skipped = 0

    for m in manifest:
        if m["ambiguous_gt"]:
            skipped += 1
            continue
        image_id = str(m["image_id"])
        gt = str(m["gt"])
        img_path = resolve_image_path(project_root, str(m["image_path"]))
        if not img_path.is_file():
            raise SystemExit(f"Không có file: {img_path}")
        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            raise SystemExit(f"OpenCV không đọc được: {img_path}")
        fd = FrameData(image_id=image_id, frame=frame_bgr, source=str(img_path))
        dets = detector.predict(fd)
        if not dets:
            if args.skip_missing_detect:
                skipped += 1
                continue
            raise SystemExit(f"Không detect được biển: {image_id} ({img_path})")
        best = max(dets, key=lambda d: d.score)
        crop = crop_plate(fd, best, margin_ratio=float(args.crop_margin_ratio))
        crop_path = out_dir / f"{image_id}.png"
        cv2.imwrite(str(crop_path), crop.crop)
        csv_rows.append(
            {
                "image_path": _rel_project(project_root, crop_path),
                "text": gt,
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_path", "text"])
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    print(f"Đã xuất {len(csv_rows)} crop → {out_dir}")
    print(f"CSV train: {out_csv} (cột image_path, text)")
    if skipped:
        print(f"Bỏ qua: {skipped} dòng (ambiguous hoặc không detect).")


if __name__ == "__main__":
    main()
