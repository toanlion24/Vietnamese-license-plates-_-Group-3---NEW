"""Sinh ảnh biển số dạng chữ in trên nền (tổng hợp) + manifest CSV có GT.

Mục đích: có tập test cố định trong repo để chạy inference/metric tái lập khi chưa đưa ảnh thật vào Git.
Không thay thế ảnh thu ngoài thực địa — trong báo cáo nên ghi rõ nguồn là synthetic.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont

# Định dạng biển VN (sau normalize): 2 số + 1 chữ + 4–5 số
DEFAULT_PLATES: tuple[str, ...] = (
    "51H12345",
    "30A56789",
    "59G11234",
    "43B67890",
    "29C24680",
    "61D13579",
    "50F99881",
    "72A45678",
    "14K10203",
    "88L99999",
    "65M44444",
    "92N12345",
    "37P88888",
    "18R24680",
    "41S13579",
)


def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/synthetic_plates"))
    p.add_argument(
        "--write-test-manifest",
        type=Path,
        default=Path("data/test_manifest.csv"),
        help="Ghi đè manifest test tại đường dẫn này (image_path tương đối project root).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Số mẫu (0 = dùng hết DEFAULT_PLATES).",
    )
    return p.parse_args()


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    plates = list(DEFAULT_PLATES)
    if args.num_samples > 0:
        plates = plates[: args.num_samples]
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    font = _find_font(42)

    manifest_rows: list[dict[str, str]] = []
    for i, gt in enumerate(plates, start=1):
        image_id = f"synth_{i:03d}"
        fname = f"{image_id}.png"
        rel_path = out_dir.relative_to(PROJECT_ROOT) / fname

        w, h = 380, 110
        img = Image.new("RGB", (w, h), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle([4, 4, w - 5, h - 5], outline=(30, 60, 120), width=3)
        text = gt
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (w - tw) // 2
        y = (h - th) // 2 - 4
        draw.text((x, y), text, fill=(10, 10, 10), font=font)

        out_path = out_dir / fname
        img.save(out_path, format="PNG")

        manifest_rows.append(
            {
                "image_id": image_id,
                "image_path": str(rel_path).replace("\\", "/"),
                "gt": gt,
                "ambiguous_gt": "false",
            }
        )

    manifest_path = (PROJECT_ROOT / args.write_test_manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_id", "image_path", "gt", "ambiguous_gt"])
        w.writeheader()
        w.writerows(manifest_rows)

    readme = out_dir / "README.md"
    readme.write_text(
        "# Ảnh biển số tổng hợp (synthetic)\n\n"
        "Ảnh được render bằng PIL (chữ in, nền sáng, viền xanh). "
        "Ground truth nằm trong `data/test_manifest.csv`.\n\n"
        "Dùng để chạy inference và metric trong repo khi không commit ảnh thật. "
        "Trong báo cáo đề tài, nên bổ sung thêm thí nghiệm trên ảnh chụp thật ngoài môi trường.\n",
        encoding="utf-8",
    )

    print(f"Đã tạo {len(manifest_rows)} ảnh trong: {out_dir}")
    print(f"Manifest test: {manifest_path}")


if __name__ == "__main__":
    main()
