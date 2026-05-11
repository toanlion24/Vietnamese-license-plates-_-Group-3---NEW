"""Fine-tune TrOCR (VisionEncoderDecoder) on crop + text pairs.

Learning rate: mặc định đọc từ ``configs/trocr/finetune_defaults.json`` (1.5e-4).
Dùng ``--learning-rate`` để ghi đè; tăng thêm nữa (vd. 2e-4–3e-4) chỉ khi theo dõi loss
và có tập validation cố định.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


def _load_json_defaults(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    raw.pop("comment", None)
    return raw


class TrocrCropDataset(Dataset):
    """Mỗi dòng: đường dẫn ảnh crop và nhãn ký tự biển số."""

    def __init__(self, rows: list[tuple[Path, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        path, text = self.rows[idx]
        image = Image.open(path).convert("RGB")
        return image, text


@dataclass
class TrOCRDataCollator:
    processor: TrOCRProcessor

    def __call__(self, batch: list[tuple[Image.Image, str]]) -> dict[str, torch.Tensor]:
        images = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        enc = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        labels = enc["labels"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc


def _parse_args() -> argparse.Namespace:
    default_cfg = PROJECT_ROOT / "configs" / "trocr" / "finetune_defaults.json"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="CSV: cột image_path (hoặc path), text (hoặc gt / label).",
    )
    p.add_argument("--val-csv", type=Path, default=None, help="Tùy chọn: CSV cùng schema cho eval.")
    p.add_argument("--model-name", type=str, default="microsoft/trocr-base-printed")
    p.add_argument("--output-dir", type=Path, default=Path("experiments/trocr_finetune"))
    p.add_argument("--model-cache-dir", type=Path, default=None)
    p.add_argument(
        "--hyperparams-json",
        type=Path,
        default=None,
        help=f"Mặc định: {default_cfg}",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Ghi đè learning rate (vd. 2e-4). Nếu bỏ trống, lấy từ hyperparams JSON.",
    )
    return p.parse_args()


def _read_pairs(csv_path: Path, project_root: Path) -> list[tuple[Path, str]]:
    import csv

    rows: list[tuple[Path, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for line_no, row in enumerate(reader, start=2):
            img_raw = (
                (row.get("image_path") or row.get("path") or row.get("file_path") or "").strip()
            )
            text = (
                (row.get("text") or row.get("gt") or row.get("label") or row.get("plate_text") or "").strip()
            )
            if not img_raw or not text:
                continue
            ip = Path(img_raw)
            if not ip.is_absolute():
                ip = project_root / ip
            if not ip.is_file():
                raise FileNotFoundError(f"Dòng {line_no}: không có file {ip}")
            rows.append((ip, text))
    if not rows:
        raise SystemExit(f"Không có cặp (ảnh, text) hợp lệ trong {csv_path}")
    return rows


def main() -> None:
    args = _parse_args()
    project_root = PROJECT_ROOT
    hp_path = args.hyperparams_json or (project_root / "configs" / "trocr" / "finetune_defaults.json")
    hp = _load_json_defaults(hp_path)

    if args.learning_rate is not None:
        hp["learning_rate"] = args.learning_rate

    train_pairs = _read_pairs(args.train_csv, project_root)
    val_pairs = _read_pairs(args.val_csv, project_root) if args.val_csv else None

    cache = str(args.model_cache_dir) if args.model_cache_dir else None
    processor = TrOCRProcessor.from_pretrained(args.model_name, cache_dir=cache)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name, cache_dir=cache)

    train_ds = TrocrCropDataset(train_pairs)
    collator = TrOCRDataCollator(processor)
    val_ds = TrocrCropDataset(val_pairs) if val_pairs else None

    train_kw = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": int(hp.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(hp.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(hp.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(hp.get("learning_rate", 1e-4)),
        "weight_decay": float(hp.get("weight_decay", 0.01)),
        "num_train_epochs": float(hp.get("num_train_epochs", 10)),
        "warmup_ratio": float(hp.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": str(hp.get("lr_scheduler_type", "linear")),
        "save_total_limit": int(hp.get("save_total_limit", 2)),
        "predict_with_generate": True,
        "logging_steps": int(hp.get("logging_steps", 20)),
        "save_strategy": str(hp.get("save_strategy", "epoch")),
        "report_to": "none",
    }
    if hp.get("fp16"):
        train_kw["fp16"] = True
    if hp.get("bf16"):
        train_kw["bf16"] = True
    if val_ds is not None:
        train_kw["eval_strategy"] = str(hp.get("eval_strategy", "epoch"))

    targs = Seq2SeqTrainingArguments(**train_kw)

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Đã lưu model + processor tại: {args.output_dir.resolve()}")
    print(f"Learning rate đã dùng: {train_kw['learning_rate']}")


if __name__ == "__main__":
    main()
