from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "bg": (248, 250, 252),
    "card": (255, 255, 255),
    "text": (15, 23, 42),
    "muted": (71, 85, 105),
    "grid": (226, 232, 240),
    "ok": (22, 163, 74),
    "missing": (220, 38, 38),
    "warn": (234, 179, 8),
    "blue": (37, 99, 235),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Buoi 4 real-result readiness diagram.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/buoi4"))
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def count_files(root: Path, patterns: tuple[str, ...]) -> int:
    total = 0
    for pattern in patterns:
        total += sum(1 for _ in root.rglob(pattern))
    return total


def status_color(ok: bool) -> tuple[int, int, int]:
    return COLORS["ok"] if ok else COLORS["missing"]


def status_text(ok: bool) -> str:
    return "OK" if ok else "THIẾU"


def draw_status_card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    detail: str,
    ok: bool,
) -> None:
    x1, y1, x2, y2 = xy
    color = status_color(ok)
    draw.rounded_rectangle(xy, radius=20, fill=COLORS["card"], outline=color, width=4)
    draw.rounded_rectangle((x1 + 24, y1 + 24, x1 + 126, y1 + 62), radius=12, fill=color)
    draw.text((x1 + 48, y1 + 30), status_text(ok), fill=(255, 255, 255), font=font(18, bold=True))
    draw.text((x1 + 24, y1 + 82), title, fill=COLORS["text"], font=font(26, bold=True))
    draw.text((x1 + 24, y1 + 118), detail, fill=COLORS["muted"], font=font(19))
    if not ok:
        draw.text((x1 + 24, y2 - 34), "Cần bổ sung để có kết quả thật", fill=COLORS["missing"], font=font(17, bold=True))


def draw_arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
    draw.line((x1, y1, x2, y2), fill=color, width=5)
    draw.polygon([(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)], fill=color)


def create_diagram(project_root: Path, output_dir: Path) -> Path:
    data_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_count = count_files(data_dir, ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"))
    label_count = count_files(data_dir, ("*.txt",))
    weight_count = count_files(project_root / "weights", ("*.pt", "*.pth", "*.onnx"))
    manifest_count = count_files(data_dir / "manifests", ("*.csv",))
    split_count = count_files(data_dir / "splits", ("*.txt",))
    demo_csv_count = count_files(project_root / "outputs" / "buoi4" / "demo", ("*.csv",))

    checks = [
        ("Ảnh test thật", f"{image_count} file ảnh trong data/", image_count > 0),
        ("Ground truth / manifest", f"{manifest_count} CSV manifest, {split_count} file split", manifest_count > 0),
        ("Checkpoint mô hình", f"{weight_count} file weight trong weights/", weight_count > 0),
        ("Nhãn/dataset phụ trợ", f"{label_count} file .txt trong data/", label_count > 0),
        ("Code đánh giá", "scripts/run_buoi4_experiments.py đã sẵn sàng", True),
        ("Demo hiện tại", f"{demo_csv_count} CSV demo đã tạo", demo_csv_count >= 2),
    ]

    img = Image.new("RGB", (1600, 1050), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 50), "Buổi 4 - Trạng thái để có kết quả thực tế", fill=COLORS["text"], font=font(44, bold=True))
    draw.text(
        (70, 110),
        "Sơ đồ này phân biệt kết quả demo hiện tại và điều kiện cần để chạy kết quả thật trên ảnh/checkpoint thật.",
        fill=COLORS["muted"],
        font=font(24),
    )

    positions = [
        (70, 190, 520, 370),
        (575, 190, 1025, 370),
        (1080, 190, 1530, 370),
        (70, 440, 520, 620),
        (575, 440, 1025, 620),
        (1080, 440, 1530, 620),
    ]
    for item, pos in zip(checks, positions):
        draw_status_card(draw, pos, item[0], item[1], item[2])

    draw.rounded_rectangle((70, 700, 1530, 950), radius=24, fill=COLORS["card"], outline=COLORS["grid"], width=2)
    draw.text((105, 735), "Pipeline chạy thật khi đủ dữ liệu", fill=COLORS["text"], font=font(30, bold=True))

    box_y = 815
    box_w, box_h = 205, 74
    steps = [
        ("Ảnh thật", COLORS["blue"]),
        ("DeepSolo\nlocalize", COLORS["blue"]),
        ("Crop +\npreprocess", COLORS["blue"]),
        ("TrOCR\nOCR", COLORS["blue"]),
        ("Eval\nCER/WER", COLORS["blue"]),
    ]
    x = 125
    for idx, (label, color) in enumerate(steps):
        draw.rounded_rectangle((x, box_y, x + box_w, box_y + box_h), radius=18, fill=COLORS["card"], outline=color, width=4)
        for line_idx, line in enumerate(label.split("\n")):
            draw.text((x + 24, box_y + 16 + line_idx * 25), line, fill=COLORS["text"], font=font(22, bold=True))
        if idx < len(steps) - 1:
            draw_arrow(draw, x + box_w, box_y + box_h // 2, x + box_w + 70, box_y + box_h // 2, COLORS["blue"])
        x += box_w + 90

    draw.text(
        (105, 910),
        "Kết luận trung thực: notebook hiện trình bày demo; để có kết quả thực tế cần thêm ảnh thật + GT text + checkpoint.",
        fill=COLORS["missing"] if image_count == 0 or weight_count == 0 or manifest_count == 0 else COLORS["ok"],
        font=font(22, bold=True),
    )

    output_path = output_dir / "real_result_readiness.png"
    img.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = create_diagram(args.project_root.resolve(), args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
