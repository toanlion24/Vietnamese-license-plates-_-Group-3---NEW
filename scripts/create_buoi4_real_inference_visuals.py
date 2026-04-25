from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "bg": (248, 250, 252),
    "card": (255, 255, 255),
    "text": (15, 23, 42),
    "muted": (71, 85, 105),
    "grid": (226, 232, 240),
    "blue": (37, 99, 235),
    "green": (22, 163, 74),
    "red": (220, 38, 38),
    "yellow": (234, 179, 8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create qualitative real inference visuals for Buoi 4.")
    parser.add_argument("--pred-json", type=Path, default=Path("outputs/buoi4/real_sample_predictions.json"))
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


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def confidence_color(confidence: float) -> tuple[int, int, int]:
    if confidence >= 0.6:
        return COLORS["green"]
    if confidence >= 0.25:
        return COLORS["yellow"]
    return COLORS["red"]


def resize_to_fit(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(max_size)
    return image


def draw_prediction_on_image(record: dict, card_size: tuple[int, int] = (560, 380)) -> Image.Image:
    card_w, card_h = card_size
    card = Image.new("RGB", card_size, COLORS["card"])
    draw = ImageDraw.Draw(card)
    source = Path(record["source"])
    image = Image.open(source).convert("RGB")
    original_w, original_h = image.size
    display = resize_to_fit(image.copy(), (card_w - 40, 250))
    img_x = (card_w - display.width) // 2
    img_y = 22
    card.paste(display, (img_x, img_y))

    scale_x = display.width / original_w
    scale_y = display.height / original_h
    bbox = record.get("bbox_xyxy")
    if bbox:
        x1, y1, x2, y2 = bbox
        scaled_bbox = (
            img_x + int(x1 * scale_x),
            img_y + int(y1 * scale_y),
            img_x + int(x2 * scale_x),
            img_y + int(y2 * scale_y),
        )
        draw.rectangle(scaled_bbox, outline=COLORS["green"], width=4)

    confidence = float(record.get("confidence", 0.0))
    latency_ms = float(record.get("latency_ms", 0.0))
    color = confidence_color(confidence)
    y_text = 292
    draw.rounded_rectangle((20, y_text, card_w - 20, card_h - 22), radius=18, fill=(248, 250, 252), outline=COLORS["grid"])
    draw.text((42, y_text + 18), record["image_id"], fill=COLORS["muted"], font=font(18, bold=True))
    draw.text((42, y_text + 48), f"OCR: {record.get('plate_text', '')}", fill=COLORS["text"], font=font(28, bold=True))
    draw.rounded_rectangle((card_w - 190, y_text + 42, card_w - 42, y_text + 80), radius=12, fill=color)
    draw.text((card_w - 174, y_text + 49), f"conf {confidence:.2f}", fill=(255, 255, 255), font=font(18, bold=True))
    draw.text((42, y_text + 88), f"Latency: {latency_ms:.1f} ms", fill=COLORS["muted"], font=font(18))
    return card


def create_grid(predictions: list[dict], output_path: Path) -> None:
    cols = 2
    rows = (len(predictions) + cols - 1) // cols
    card_w, card_h = 560, 380
    margin = 60
    gap = 35
    header_h = 160
    width = margin * 2 + cols * card_w + (cols - 1) * gap
    height = header_h + margin + rows * card_h + (rows - 1) * gap
    img = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((margin, 42), "Kết quả inference thật trên ảnh mẫu", fill=COLORS["text"], font=font(42, bold=True))
    draw.text(
        (margin, 100),
        "YOLOv8 checkpoint trong weights/ + EasyOCR. Đây là kết quả định tính vì chưa có GT text để tính accuracy.",
        fill=COLORS["muted"],
        font=font(22),
    )

    for idx, record in enumerate(predictions):
        row, col = divmod(idx, cols)
        x = margin + col * (card_w + gap)
        y = header_h + row * (card_h + gap)
        card = draw_prediction_on_image(record, (card_w, card_h))
        img.paste(card, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def create_real_summary(predictions: list[dict], output_path: Path) -> None:
    img = Image.new("RGB", (1500, 820), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    draw.text((70, 50), "Sơ đồ kết quả thực tế hiện có", fill=COLORS["text"], font=font(44, bold=True))
    draw.text(
        (70, 110),
        "Đã chạy inference trên ảnh trong repo; phần còn thiếu là GT text để tính CER/WER/plate accuracy thật.",
        fill=COLORS["muted"],
        font=font(24),
    )

    avg_conf = sum(float(row.get("confidence", 0.0)) for row in predictions) / max(1, len(predictions))
    avg_latency = sum(float(row.get("latency_ms", 0.0)) for row in predictions) / max(1, len(predictions))
    non_empty = sum(1 for row in predictions if row.get("plate_text"))

    cards = [
        ("Ảnh đã chạy", f"{len(predictions)} ảnh mẫu", COLORS["blue"]),
        ("Có OCR output", f"{non_empty}/{len(predictions)} ảnh", COLORS["green"]),
        ("Confidence TB", f"{avg_conf:.2f}", confidence_color(avg_conf)),
        ("Latency TB", f"{avg_latency:.1f} ms", COLORS["yellow"]),
    ]
    x = 70
    for title, value, color in cards:
        draw.rounded_rectangle((x, 190, x + 320, 365), radius=22, fill=COLORS["card"], outline=COLORS["grid"], width=2)
        draw.text((x + 28, 220), title, fill=COLORS["muted"], font=font(23, bold=True))
        draw.text((x + 28, 270), value, fill=color, font=font(38, bold=True))
        x += 360

    draw.rounded_rectangle((70, 440, 1430, 720), radius=24, fill=COLORS["card"], outline=COLORS["grid"], width=2)
    draw.text((105, 475), "Luồng kết quả thật", fill=COLORS["text"], font=font(30, bold=True))
    steps = ["Ảnh trong\ndata/debug_aug", "YOLOv8\nbbox", "Crop +\npreprocess", "EasyOCR\ntext", "Ảnh kết quả\n+ JSON"]
    step_x = 115
    for idx, step in enumerate(steps):
        draw.rounded_rectangle((step_x, 555, step_x + 205, 635), radius=18, fill=COLORS["card"], outline=COLORS["blue"], width=4)
        for line_idx, line in enumerate(step.split("\n")):
            draw.text((step_x + 22, 574 + line_idx * 25), line, fill=COLORS["text"], font=font(20, bold=True))
        if idx < len(steps) - 1:
            draw.line((step_x + 205, 595, step_x + 270, 595), fill=COLORS["blue"], width=5)
            draw.polygon([(step_x + 270, 595), (step_x + 252, 585), (step_x + 252, 605)], fill=COLORS["blue"])
        step_x += 270
    draw.text((105, 675), "Để có số accuracy thật: thêm manifest image_id + text_gt rồi chạy lại evaluator.", fill=COLORS["red"], font=font(22, bold=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.pred_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    create_grid(predictions, args.output_dir / "real_inference_grid.png")
    create_real_summary(predictions, args.output_dir / "real_inference_summary.png")
    print(args.output_dir / "real_inference_summary.png")
    print(args.output_dir / "real_inference_grid.png")


if __name__ == "__main__":
    main()
