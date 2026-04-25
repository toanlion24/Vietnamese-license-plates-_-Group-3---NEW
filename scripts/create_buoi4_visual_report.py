from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "bg": (248, 250, 252),
    "card": (255, 255, 255),
    "text": (15, 23, 42),
    "muted": (71, 85, 105),
    "grid": (226, 232, 240),
    "a": (37, 99, 235),
    "b": (22, 163, 74),
    "bad": (220, 38, 38),
    "warn": (234, 179, 8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PNG visuals for Buoi 4 A/B notebook report.")
    parser.add_argument("--metrics-json", type=Path, default=Path("reports/buoi4_demo_ab_metrics.json"))
    parser.add_argument(
        "--config-a-csv",
        type=Path,
        default=Path("outputs/buoi4/demo/deepsolo_e2e_predictions.csv"),
    )
    parser.add_argument(
        "--config-b-csv",
        type=Path,
        default=Path("outputs/buoi4/demo/deepsolo_trocr_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/buoi4"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


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


def rounded_card(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(xy, radius=22, fill=COLORS["card"], outline=COLORS["grid"], width=2)


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    value: str,
    subtitle: str,
    color: tuple[int, int, int],
) -> None:
    rounded_card(draw, xy)
    x1, y1, x2, _ = xy
    draw.text((x1 + 28, y1 + 24), title, fill=COLORS["muted"], font=font(24, bold=True))
    draw.text((x1 + 28, y1 + 66), value, fill=color, font=font(44, bold=True))
    draw.text((x1 + 28, y1 + 124), subtitle, fill=COLORS["text"], font=font(20))
    draw.line((x1 + 28, y1 + 154, x2 - 28, y1 + 154), fill=COLORS["grid"], width=2)
    draw.text((x1 + 28, y1 + 170), "Demo/smoke-test, chưa phải kết quả mô hình thật", fill=COLORS["muted"], font=font(16))


def create_overview(metrics: dict, output_path: Path) -> None:
    img = Image.new("RGB", (1500, 900), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 55), "Buổi 4 - Tổng quan thực nghiệm A/B", fill=COLORS["text"], font=font(44, bold=True))
    draw.text(
        (70, 115),
        "DeepSolo end-to-end vs DeepSolo + TrOCR trên bộ demo 8 mẫu",
        fill=COLORS["muted"],
        font=font(24),
    )

    a = metrics["config_a"]
    b = metrics["config_b"]
    draw_metric_card(
        draw,
        (70, 190, 710, 420),
        "A - DeepSolo end-to-end",
        f"{a['plate_accuracy'] * 100:.1f}% plate accuracy",
        f"CER {a['cer']:.4f} | WER {a['wer']:.4f} | latency {a['mean_latency_ms']:.1f} ms",
        COLORS["a"],
    )
    draw_metric_card(
        draw,
        (790, 190, 1430, 420),
        "B - DeepSolo + TrOCR",
        f"{b['plate_accuracy'] * 100:.1f}% plate accuracy",
        f"CER {b['cer']:.4f} | WER {b['wer']:.4f} | latency {b['mean_latency_ms']:.1f} ms",
        COLORS["b"],
    )

    rounded_card(draw, (70, 485, 1430, 810))
    draw.text((105, 520), "Kết luận để trình bày", fill=COLORS["text"], font=font(30, bold=True))
    bullets = [
        "Cấu hình B đọc đúng nhiều biển số hơn trong demo: 87.5% so với 62.5%.",
        "Cấu hình B có CER/WER thấp hơn, nghĩa là lỗi ký tự và lỗi token ít hơn.",
        "Đổi lại, cấu hình B chậm hơn do phải chạy 2 stage: localize rồi OCR.",
        "Hướng Buổi 5: ưu tiên B nếu mục tiêu là độ chính xác, sau đó tối ưu crop và latency.",
    ]
    y = 575
    for item in bullets:
        draw.ellipse((110, y + 8, 124, y + 22), fill=COLORS["b"])
        draw.text((145, y), item, fill=COLORS["text"], font=font(24))
        y += 52

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def create_bar_chart(metrics: dict, output_path: Path) -> None:
    img = Image.new("RGB", (1400, 780), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 45), "So sánh metric chính", fill=COLORS["text"], font=font(42, bold=True))
    draw.text((70, 102), "Plate accuracy càng cao càng tốt; CER/WER càng thấp càng tốt.", fill=COLORS["muted"], font=font(22))

    chart_x, chart_y, chart_w, chart_h = 120, 190, 1160, 460
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=COLORS["grid"], width=3)
    metrics_to_plot = [
        ("Plate accuracy", metrics["config_a"]["plate_accuracy"], metrics["config_b"]["plate_accuracy"]),
        ("CER", metrics["config_a"]["cer"], metrics["config_b"]["cer"]),
        ("WER", metrics["config_a"]["wer"], metrics["config_b"]["wer"]),
    ]
    max_value = 1.0
    group_gap = 300
    bar_w = 72
    for idx, (label, value_a, value_b) in enumerate(metrics_to_plot):
        base_x = chart_x + 115 + idx * group_gap
        for offset, value, color, name in [(0, value_a, COLORS["a"], "A"), (95, value_b, COLORS["b"], "B")]:
            bar_h = int((value / max_value) * chart_h)
            x1 = base_x + offset
            y1 = chart_y + chart_h - bar_h
            draw.rounded_rectangle((x1, y1, x1 + bar_w, chart_y + chart_h), radius=10, fill=color)
            draw.text((x1 - 4, y1 - 34), f"{value:.3f}", fill=COLORS["text"], font=font(20, bold=True))
            draw.text((x1 + 24, chart_y + chart_h + 18), name, fill=COLORS["muted"], font=font(22, bold=True))
        draw.text((base_x - 35, chart_y + chart_h + 58), label, fill=COLORS["text"], font=font(22, bold=True))

    draw.rounded_rectangle((985, 108, 1280, 168), radius=16, fill=COLORS["card"], outline=COLORS["grid"])
    draw.rectangle((1010, 126, 1035, 151), fill=COLORS["a"])
    draw.text((1048, 122), "A - DeepSolo", fill=COLORS["text"], font=font(20))
    draw.rectangle((1170, 126, 1195, 151), fill=COLORS["b"])
    draw.text((1208, 122), "B - TrOCR", fill=COLORS["text"], font=font(20))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def create_latency_chart(metrics: dict, output_path: Path) -> None:
    img = Image.new("RGB", (1200, 650), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 45), "Đổi độ chính xác lấy latency", fill=COLORS["text"], font=font(42, bold=True))
    draw.text((70, 102), "B chính xác hơn nhưng chậm hơn vì pipeline 2 stage.", fill=COLORS["muted"], font=font(22))

    a_latency = float(metrics["config_a"]["mean_latency_ms"])
    b_latency = float(metrics["config_b"]["mean_latency_ms"])
    max_latency = max(a_latency, b_latency) * 1.2
    labels = [("A - DeepSolo", a_latency, COLORS["a"]), ("B - DeepSolo + TrOCR", b_latency, COLORS["b"])]
    y = 220
    for label, value, color in labels:
        draw.text((90, y), label, fill=COLORS["text"], font=font(26, bold=True))
        bar_w = int((value / max_latency) * 760)
        draw.rounded_rectangle((390, y - 5, 390 + bar_w, y + 42), radius=14, fill=color)
        draw.text((390 + bar_w + 24, y + 2), f"{value:.1f} ms", fill=COLORS["text"], font=font(26, bold=True))
        y += 120

    draw.text((90, 500), "Thông điệp khi bảo vệ: tuy B chậm hơn, nó có độ chính xác cao hơn và dễ debug hơn.", fill=COLORS["muted"], font=font(24))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def create_pipeline_diagram(output_path: Path) -> None:
    img = Image.new("RGB", (1500, 760), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 50), "Sơ đồ hai cấu hình Buổi 4", fill=COLORS["text"], font=font(42, bold=True))

    def box(x: int, y: int, w: int, h: int, text: str, color: tuple[int, int, int]) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=20, fill=COLORS["card"], outline=color, width=4)
        lines = text.split("\n")
        total_h = len(lines) * 28
        for idx, line in enumerate(lines):
            draw.text((x + 24, y + h // 2 - total_h // 2 + idx * 32), line, fill=COLORS["text"], font=font(24, bold=True))

    def arrow(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        draw.line((x1, y1, x2, y2), fill=color, width=5)
        draw.polygon([(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)], fill=color)

    draw.text((90, 145), "A - DeepSolo end-to-end", fill=COLORS["a"], font=font(30, bold=True))
    box(90, 200, 230, 105, "Ảnh/frame", COLORS["a"])
    box(420, 200, 310, 105, "DeepSolo\nspotting", COLORS["a"])
    box(830, 200, 310, 105, "Polygon/BBox\n+ text", COLORS["a"])
    box(1230, 200, 190, 105, "Eval", COLORS["a"])
    arrow(320, 252, 420, 252, COLORS["a"])
    arrow(730, 252, 830, 252, COLORS["a"])
    arrow(1140, 252, 1230, 252, COLORS["a"])

    draw.text((90, 405), "B - DeepSolo + TrOCR", fill=COLORS["b"], font=font(30, bold=True))
    box(90, 460, 230, 105, "Ảnh/frame", COLORS["b"])
    box(390, 460, 240, 105, "DeepSolo\nlocalize", COLORS["b"])
    box(700, 460, 210, 105, "Crop +\npreprocess", COLORS["b"])
    box(980, 460, 210, 105, "TrOCR\nOCR", COLORS["b"])
    box(1260, 460, 160, 105, "Eval", COLORS["b"])
    arrow(320, 512, 390, 512, COLORS["b"])
    arrow(630, 512, 700, 512, COLORS["b"])
    arrow(910, 512, 980, 512, COLORS["b"])
    arrow(1190, 512, 1260, 512, COLORS["b"])

    draw.text((90, 655), "Điểm cần nói với thầy: A gọn hơn; B dễ debug và hiện tại chính xác hơn trong demo.", fill=COLORS["muted"], font=font(24))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def create_sample_outcomes(pred_a: list[dict[str, str]], pred_b: list[dict[str, str]], output_path: Path) -> None:
    img = Image.new("RGB", (1500, 900), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 45), "Đúng/sai theo từng mẫu demo", fill=COLORS["text"], font=font(42, bold=True))
    draw.text((70, 102), "Màu xanh = đọc đúng hoàn toàn; màu đỏ = sai sau normalize/demo.", fill=COLORS["muted"], font=font(22))

    pred_b_by_id = {row["image_id"]: row for row in pred_b}
    x_cols = [80, 260, 505, 745, 985, 1225]
    headers = ["Image ID", "GT", "Pred A", "A", "Pred B", "B"]
    for x, header in zip(x_cols, headers):
        draw.text((x, 170), header, fill=COLORS["text"], font=font(24, bold=True))
    draw.line((70, 215, 1430, 215), fill=COLORS["grid"], width=3)

    y = 240
    for row_a in pred_a:
        row_b = pred_b_by_id.get(row_a["image_id"], {})
        a_ok = row_a["gt"] == row_a["pred"]
        b_ok = row_a["gt"] == row_b.get("pred", "")
        values = [row_a["image_id"], row_a["gt"], row_a["pred"], "OK" if a_ok else "SAI", row_b.get("pred", ""), "OK" if b_ok else "SAI"]
        for x, value in zip(x_cols, values):
            if value in {"OK", "SAI"}:
                color = COLORS["b"] if value == "OK" else COLORS["bad"]
                draw.rounded_rectangle((x - 12, y - 6, x + 78, y + 34), radius=10, fill=color)
                draw.text((x + 8, y), value, fill=(255, 255, 255), font=font(20, bold=True))
            else:
                draw.text((x, y), value, fill=COLORS["text"], font=font(22))
        y += 68

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    args = parse_args()
    metrics = load_json(args.metrics_json)
    pred_a = load_csv(args.config_a_csv)
    pred_b = load_csv(args.config_b_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    create_overview(metrics, args.output_dir / "overview.png")
    create_bar_chart(metrics, args.output_dir / "metrics_comparison.png")
    create_latency_chart(metrics, args.output_dir / "latency_comparison.png")
    create_pipeline_diagram(args.output_dir / "pipeline_ab.png")
    create_sample_outcomes(pred_a, pred_b, args.output_dir / "sample_outcomes.png")

    manifest = {
        "overview": str(args.output_dir / "overview.png"),
        "metrics_comparison": str(args.output_dir / "metrics_comparison.png"),
        "latency_comparison": str(args.output_dir / "latency_comparison.png"),
        "pipeline_ab": str(args.output_dir / "pipeline_ab.png"),
        "sample_outcomes": str(args.output_dir / "sample_outcomes.png"),
    }
    (args.output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
