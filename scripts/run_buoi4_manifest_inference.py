"""Run Buổi 4 A/B evaluation from a fixed manifest: YOLOv8 + OCR backends or imported CSV.

Repo không nhúng DeepSolo. Mặc định:
  - Cấu hình A: YOLOv8 (hoặc ``dummy``) + EasyOCR thật.
  - Cấu hình B: YOLOv8 (hoặc ``dummy``) + TrOCR thật.

Chỉ khi truyền ``--ocr-dummy`` thì hai nhánh dùng ``DummyOcr`` (smoke test), không phản ánh chất lượng OCR.

Khi đã có DeepSolo export CSV đúng schema, dùng ``--config-a-from-csv`` / ``--config-b-from-csv``
để nhập prediction thay cho inference tương ứng. Ground truth luôn lấy từ manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.error_labels import classify_plate_error
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.detector.base import DummyCenterDetector
from src.ocr.easyocr_adapter import EasyOcrAdapter
from src.ocr.ensemble_easy_trocr import EasyTrocrEnsembleOcr
from src.ocr.trocr_adapter import TrOcrAdapter
from src.ocr.base import DummyOcr
from src.pipeline.detailed_plate_infer import infer_plate_detailed
from src.utils.types import FrameData


def project_rel(project_root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/test_manifest.csv"))
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument(
        "--output-a-csv",
        type=Path,
        default=Path("outputs/buoi4/deepsolo_e2e_predictions.csv"),
        help="Config A predictions (tên file giữ theo Buổi 4).",
    )
    p.add_argument(
        "--output-b-csv",
        type=Path,
        default=Path("outputs/buoi4/deepsolo_trocr_predictions.csv"),
    )
    p.add_argument("--detector-backend", choices=["yolov8", "dummy"], default="yolov8")
    p.add_argument("--detector-model", type=Path, default=Path("weights/yolov8_license_plate.pt"))
    p.add_argument("--detector-conf", type=float, default=0.25)
    p.add_argument("--ocr-gpu", action="store_true")
    p.add_argument(
        "--ocr-dummy",
        action="store_true",
        help="Dùng DummyOcr (chỉ để kiểm tra dây nối). Mặc định luôn chạy EasyOCR/TrOCR thật.",
    )
    p.add_argument("--trocr-model", type=str, default="microsoft/trocr-base-printed")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model-cache-dir", type=Path, default=None)
    p.add_argument("--bad-crop-det-conf", type=float, default=0.4)

    p.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=0.0,
        help="Mở rộng bbox crop theo tỉ lệ (vd. 0.08 = thêm ~8%% mỗi phía).",
    )
    p.add_argument(
        "--preprocess-clahe",
        action="store_true",
        help="Dùng CLAHE trên ảnh xám trước khi resize (thường giúp tương phản).",
    )
    p.add_argument(
        "--aggressive-postprocess",
        action="store_true",
        help="Bật sửa ký tự hay nhầm ở phần số sau chữ tỉnh (kiểm tra trên val trước).",
    )
    p.add_argument("--yolo-iou", type=float, default=0.45, help="NMS IoU cho ultralytics YOLO.predict.")
    p.add_argument("--yolo-imgsz", type=int, default=640, help="Kích thước infer YOLO (ảnh vuông).")
    p.add_argument(
        "--ensemble-b",
        action="store_true",
        help="Nhánh B dùng EasyOCR+TrOCR ensemble (chọn theo regex biển VN + confidence).",
    )

    p.add_argument(
        "--config-a-from-csv",
        type=Path,
        default=None,
        help="Nếu có: không chạy inference A, đọc pred từ CSV (schema như output inference).",
    )
    p.add_argument(
        "--config-b-from-csv",
        type=Path,
        default=None,
        help="Nếu có: không chạy inference B, đọc pred từ CSV.",
    )

    p.add_argument(
        "--experiment-note",
        type=str,
        default="",
        help="Ghi chú thêm khi chạy --run-metrics (ví dụ: checkpoint, GPU).",
    )
    p.add_argument("--run-metrics", action="store_true", help="Chạy luôn run_buoi4_experiments.py.")
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("reports/buoi4_ab_metrics.json"),
    )
    p.add_argument(
        "--report-md",
        type=Path,
        default=Path("docs/buoi-4-ket-qua-thuc-nghiem-deepsolo-trocr.md"),
    )
    p.add_argument("--export-hard-cases", action="store_true", help="Sinh Markdown hard cases (xem --hard-cases-md).")
    p.add_argument(
        "--hard-cases-md",
        type=Path,
        default=Path("reports/buoi4_hard_cases.md"),
        help="Đích cho export hard cases khi bật --export-hard-cases.",
    )
    return p.parse_args()


def _infer_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "crop_margin_ratio": float(args.crop_margin_ratio),
        "preprocess_clahe": bool(args.preprocess_clahe),
        "aggressive_postprocess": bool(args.aggressive_postprocess),
    }


def _first_nonempty(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def _parse_ambiguous(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "y", "ambiguous")


def load_manifest(path: Path) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
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
            amb_raw = str(row.get("ambiguous_gt", "")).strip()
            rows_out.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "gt": gt,
                    "ambiguous_gt": _parse_ambiguous(amb_raw),
                }
            )
    return rows_out


def resolve_image_path(project_root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def bbox_to_csv(bbox: tuple[int, int, int, int] | None) -> str:
    if not bbox:
        return ""
    return ",".join(str(x) for x in bbox)


def _parse_bbox(raw: str) -> tuple[int, int, int, int] | None:
    raw = raw.strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        return None
    return int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))


def load_prediction_import(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fields = reader.fieldnames or ()
        for row in reader:
            iid = _first_nonempty(row, ("image_id", "id", "filename"))
            if not iid:
                continue
            out[iid] = {k: (row.get(k) or "") for k in fields}
    return out


def _build_detector(args: argparse.Namespace):
    if args.detector_backend == "dummy":
        return DummyCenterDetector()
    return YoloV8PlateDetector(
        model_path=args.detector_model,
        conf_threshold=args.detector_conf,
        iou=float(args.yolo_iou),
        imgsz=int(args.yolo_imgsz),
    )


def _build_ocr_easy(args: argparse.Namespace):
    if args.ocr_dummy:
        return DummyOcr()
    return EasyOcrAdapter(gpu=args.ocr_gpu)


def _build_ocr_trocr(args: argparse.Namespace):
    if args.ocr_dummy:
        return DummyOcr()
    return TrOcrAdapter(
        model_name=args.trocr_model,
        device=args.device,
        cache_dir=str(args.model_cache_dir) if args.model_cache_dir else None,
    )


def _build_ocr_ensemble_b(args: argparse.Namespace):
    if args.ocr_dummy:
        return DummyOcr()
    easy = EasyOcrAdapter(gpu=args.ocr_gpu)
    trocr = TrOcrAdapter(
        model_name=args.trocr_model,
        device=args.device,
        cache_dir=str(args.model_cache_dir) if args.model_cache_dir else None,
    )
    return EasyTrocrEnsembleOcr(easy, trocr, aggressive_post=bool(args.aggressive_postprocess))


def _row_from_import(
    *,
    gt: str,
    image_id: str,
    pack: dict[str, str],
    ambiguous_gt: bool,
    bad_crop_det_conf: float,
) -> dict[str, str]:
    pred = _first_nonempty(pack, ("pred", "text_pred", "plate_text", "prediction"))
    score_s = pack.get("score", "") or pack.get("confidence", "")
    latency_s = pack.get("latency_ms", "") or pack.get("latency", "")
    bbox_raw = (pack.get("bbox_xyxy", "") or pack.get("bbox", "")).strip()
    bbox = _parse_bbox(bbox_raw)
    ocr_raw_optional = (pack.get("ocr_text_raw", "") or pack.get("text_raw", "")).strip()

    try:
        det_score = float(score_s)
    except ValueError:
        det_score = 1.0 if bbox is not None else 0.0

    detect_hit = bbox is not None

    err = classify_plate_error(
        gt=gt,
        pred=pred,
        detect_hit=detect_hit,
        ambiguous_gt=ambiguous_gt,
        ocr_raw_norm=ocr_raw_optional,
        pred_before_repair="",
        det_score=det_score,
        bad_crop_det_conf=bad_crop_det_conf,
    )
    return {
        "image_id": image_id,
        "gt": gt,
        "pred": pred,
        "score": score_s if score_s else (f"{det_score:.4f}" if bbox is not None else ""),
        "latency_ms": latency_s if latency_s else "",
        "bbox_xyxy": bbox_raw,
        "error_type": err,
    }


def _row_from_infer(
    *,
    gt: str,
    ambiguous_gt: bool,
    detail,
    bad_crop_det_conf: float,
) -> dict[str, str]:
    bbox = detail.bbox_xyxy
    detect_hit = bbox is not None
    err = classify_plate_error(
        gt=gt,
        pred=detail.plate_text,
        detect_hit=detect_hit,
        ambiguous_gt=ambiguous_gt,
        ocr_raw_norm=detail.ocr_text_raw,
        pred_before_repair=detail.text_before_repair,
        det_score=detail.det_score,
        bad_crop_det_conf=bad_crop_det_conf,
    )
    return {
        "image_id": detail.image_id,
        "gt": gt,
        "pred": detail.plate_text,
        "score": f"{detail.confidence:.4f}" if detail.confidence else "",
        "latency_ms": f"{detail.latency_ms:.4f}",
        "bbox_xyxy": bbox_to_csv(bbox),
        "error_type": err,
    }


def _write_prediction_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "gt", "pred", "score", "latency_ms", "bbox_xyxy", "error_type"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest_path = (project_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest

    if not manifest_path.is_file():
        raise SystemExit(f"Không tìm thấy manifest: {manifest_path}")

    manifest_rows = load_manifest(manifest_path)
    if not manifest_rows:
        raise SystemExit(
            f"Manifest rỗng: {manifest_path}. Thêm ít nhất một dòng (xem data/test_manifest.example.csv)."
        )

    note_default = (
        "Cấu hình A: YOLOv8 + EasyOCR. Cấu hình B: YOLOv8 + TrOCR (hoặc ensemble nếu --ensemble-b). "
        "Có thể thay bằng DeepSolo qua flags --config-*-from-csv. "
        "GT lấy từ manifest; pred từ inference hoặc CSV import."
    )
    if args.ensemble_b:
        note_default += " Nhánh B dùng EasyOCR+TrOCR ensemble."
    if args.crop_margin_ratio > 0 or args.preprocess_clahe or args.aggressive_postprocess:
        note_default += (
            f" Crop margin={args.crop_margin_ratio}, CLAHE={args.preprocess_clahe}, "
            f"aggressive_post={args.aggressive_postprocess}."
        )
    experiment_note = args.experiment_note.strip() or note_default

    import_a = load_prediction_import(args.config_a_from_csv) if args.config_a_from_csv else {}
    import_b = load_prediction_import(args.config_b_from_csv) if args.config_b_from_csv else {}

    detector = _build_detector(args)
    ocr_easy = _build_ocr_easy(args)
    ocr_b = _build_ocr_ensemble_b(args) if args.ensemble_b else _build_ocr_trocr(args)
    infer_kw = _infer_kwargs(args)

    rows_a: list[dict[str, str]] = []
    rows_b: list[dict[str, str]] = []

    for mrow in manifest_rows:
        image_id = str(mrow["image_id"])
        gt = str(mrow["gt"])
        ambiguous_gt = bool(mrow["ambiguous_gt"])
        img_path = resolve_image_path(project_root, str(mrow["image_path"]))

        if image_id in import_a:
            rows_a.append(
                _row_from_import(
                    gt=gt,
                    image_id=image_id,
                    pack=import_a[image_id],
                    ambiguous_gt=ambiguous_gt,
                    bad_crop_det_conf=args.bad_crop_det_conf,
                )
            )
        else:
            if not img_path.is_file():
                raise SystemExit(f"Không đọc được ảnh cho {image_id}: {img_path}")
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise SystemExit(f"OpenCV không đọc được file: {img_path}")
            frame_data = FrameData(image_id=image_id, frame=frame, source=str(img_path))
            det_a = infer_plate_detailed(detector, ocr_easy, frame_data, **infer_kw)
            rows_a.append(
                _row_from_infer(
                    gt=gt,
                    ambiguous_gt=ambiguous_gt,
                    detail=det_a,
                    bad_crop_det_conf=args.bad_crop_det_conf,
                )
            )

        if image_id in import_b:
            rows_b.append(
                _row_from_import(
                    gt=gt,
                    image_id=image_id,
                    pack=import_b[image_id],
                    ambiguous_gt=ambiguous_gt,
                    bad_crop_det_conf=args.bad_crop_det_conf,
                )
            )
        else:
            if not img_path.is_file():
                raise SystemExit(f"Không đọc được ảnh cho {image_id}: {img_path}")
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise SystemExit(f"OpenCV không đọc được file: {img_path}")
            frame_data = FrameData(image_id=image_id, frame=frame, source=str(img_path))
            det_b = infer_plate_detailed(detector, ocr_b, frame_data, **infer_kw)
            rows_b.append(
                _row_from_infer(
                    gt=gt,
                    ambiguous_gt=ambiguous_gt,
                    detail=det_b,
                    bad_crop_det_conf=args.bad_crop_det_conf,
                )
            )

    out_a = (project_root / args.output_a_csv).resolve() if not args.output_a_csv.is_absolute() else args.output_a_csv
    out_b = (project_root / args.output_b_csv).resolve() if not args.output_b_csv.is_absolute() else args.output_b_csv
    _write_prediction_csv(out_a, rows_a)
    _write_prediction_csv(out_b, rows_b)

    summary = {
        "manifest": str(manifest_path),
        "num_samples": len(rows_a),
        "config_a_csv": str(out_a),
        "config_b_csv": str(out_b),
        "import_a": str(args.config_a_from_csv) if args.config_a_from_csv else None,
        "import_b": str(args.config_b_from_csv) if args.config_b_from_csv else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.run_metrics:
        metrics_json = project_root / args.metrics_json if not args.metrics_json.is_absolute() else args.metrics_json
        report_md = project_root / args.report_md if not args.report_md.is_absolute() else args.report_md

        def _rel(p: Path) -> str:
            return project_rel(project_root, p)

        cmd = [
            sys.executable,
            _rel(project_root / "scripts" / "run_buoi4_experiments.py"),
            "--config-a-csv",
            _rel(out_a),
            "--config-b-csv",
            _rel(out_b),
            "--metrics-json",
            _rel(metrics_json),
            "--report-md",
            _rel(report_md),
            "--experiment-note",
            experiment_note,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(project_root), check=True)

    if args.export_hard_cases:
        hc_script = project_root / "scripts" / "export_buoi4_hard_cases.py"
        hc_out = project_root / args.hard_cases_md if not args.hard_cases_md.is_absolute() else args.hard_cases_md

        def _rel(p: Path) -> str:
            return project_rel(project_root, p)

        cmd2 = [
            sys.executable,
            _rel(hc_script),
            "--config-a-csv",
            _rel(out_a),
            "--config-b-csv",
            _rel(out_b),
            "--output-md",
            _rel(hc_out),
            "--limit",
            "20",
        ]
        print("Running:", " ".join(cmd2))
        subprocess.run(cmd2, cwd=str(project_root), check=True)


if __name__ == "__main__":
    main()
