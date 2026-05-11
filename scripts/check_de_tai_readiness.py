"""Kiểm tra nhanh các điều kiện 'sẵn sàng' cho đề tài (không thay thế báo cáo hay review thủ công)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"Project root: {PROJECT_ROOT}\n")

    py = sys.executable
    r = subprocess.run([py, "-m", "compileall", "src", "scripts"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        _ok("compileall src scripts")
    else:
        _fail("compileall src scripts")
        print(r.stdout)
        print(r.stderr)

    weights = PROJECT_ROOT / "weights" / "yolov8_license_plate.pt"
    if weights.is_file():
        _ok(f"weight mặc định tồn tại: {weights.relative_to(PROJECT_ROOT)}")
    else:
        _warn(
            "chưa có weights/yolov8_license_plate.pt — cần huấn luyện/copy checkpoint để inference thật."
        )

    tmanifest = PROJECT_ROOT / "data" / "test_manifest.csv"
    if tmanifest.is_file():
        text = tmanifest.read_text(encoding="utf-8-sig").strip()
        n_data_lines = max(0, len([ln for ln in text.splitlines()[1:] if ln.strip()]))
        if n_data_lines > 0:
            _ok(f"data/test_manifest.csv có {n_data_lines} dòng dữ liệu")
        else:
            _warn("data/test_manifest.csv chỉ có header — thêm image_path + gt trước khi đánh giá thật.")
    else:
        _warn("chưa có data/test_manifest.csv")

    demo_a = PROJECT_ROOT / "outputs" / "buoi4" / "demo" / "deepsolo_e2e_predictions.csv"
    if demo_a.is_file():
        _ok("có CSV demo Buổi 4 (outputs/buoi4/demo/)")
    else:
        _warn("chưa chạy create_buoi4_demo_predictions hoặc chưa có demo CSV")

    checklist = PROJECT_ROOT / "docs" / "KIEM_TRA_DE_TAI_VA_HOI_DONG.md"
    if checklist.is_file():
        _ok("có docs/KIEM_TRA_DE_TAI_VA_HOI_DONG.md — đọc trước khi bảo vệ")
    else:
        _warn("thiếu docs/KIEM_TRA_DE_TAI_VA_HOI_DONG.md")

    print("\nGợi ý tiếp theo: điền test set, chạy run_buoi4_manifest_inference.py hoặc nhập CSV DeepSolo, rồi run_buoi4_experiments.py.")


if __name__ == "__main__":
    main()
