"""Fine-tune YOLOv8 for Vietnam license plate detection.

Usage:
    # From project root (D:\ComputerVisionNew):
    python scripts/train_detector.py                                      # defaults
    python scripts/train_detector.py --epochs 30 --batch 16 --imgsz 640  # custom
    python scripts/train_detector.py --resume                             # resume last run
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

# ── project root so scripts/ imports work ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(f"[train_detector] ultralytics not installed: {exc}") from exc


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = PROJECT_ROOT / "configs/detector/yolov8n_finetune.yaml"
DEFAULT_DATA   = PROJECT_ROOT / "data/data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for Vietnam license plate detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_argument_group("Model / Data")
    g.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YOLO training config YAML",
    )
    g.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA,
        help="Dataset YAML (images/labels root)",
    )
    g.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pretrained weights (e.g. yolov8n.pt, yolov8s.pt). "
             "Overrides model from --config.",
    )
    g.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last interrupted training run.",
    )

    g = parser.add_argument_group("Core training")
    g.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    g.add_argument("--batch", type=int, default=None, help="Override batch size")
    g.add_argument("--imgsz", type=int, default=None, help="Override input image size")
    g.add_argument("--patience", type=int, default=None, help="Early-stop patience")

    g = parser.add_argument_group("Hardware")
    g.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '0', 'cpu', or '' (auto-select GPU if available).",
    )

    g = parser.add_argument_group("Output")
    g.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Override output project directory.",
    )
    g.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override experiment name.",
    )
    g.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing experiment folder.",
    )
    g.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cpu_count() -> int:
    import os
    return min(os.cpu_count() or 4, 8)


def check_dataset(data_yaml: Path) -> bool:
    """Validate that data.yaml paths and image files exist."""
    cfg = _load_yaml(data_yaml)
    ok = True
    path_root = Path(cfg.get("path", ".")).resolve()

    for split in ("train", "val", "test"):
        split_file = cfg.get(split)
        if not split_file:
            continue
        split_path = path_root / split_file
        if not split_path.exists():
            print(f"[WARN] Split file not found: {split_path}")
            ok = False
            continue
        # Sample-check that images exist
        lines = split_path.read_text(encoding="utf-8").strip().splitlines()
        missing = 0
        checked = 0
        for line in lines[:20]:  # sample 20
            img_path = Path(line.strip())
            if img_path.is_absolute():
                check_path = img_path
            else:
                check_path = path_root / img_path
            if not check_path.exists():
                missing += 1
            checked += 1
        if missing:
            print(f"[WARN] {split}: {missing}/{checked} sampled images missing in {split_path.name}")
            # Only fail if more than 50% of sampled images are missing
            if missing / checked > 0.5:
                ok = False
    return ok


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── resolve config ─────────────────────────────────────────────────────────
    if args.config.exists():
        config = _load_yaml(args.config)
        print(f"[train_detector] Loaded config: {args.config}")
    else:
        print(f"[WARN] Config not found: {args.config}, using CLI defaults only.")
        config = {}

    if not args.data_yaml.exists():
        raise FileNotFoundError(f"[train_detector] Dataset YAML not found: {args.data_yaml}")

    print(f"[train_detector] Dataset YAML: {args.data_yaml}")
    if not check_dataset(args.data_yaml):
        print("[train_detector] Dataset check FAILED — fix paths before training.")
        sys.exit(1)
    print("[train_detector] Dataset check PASSED.")

    # ── device ────────────────────────────────────────────────────────────────
    device = args.device.strip()
    if device == "":
        device = "0"  # auto-select GPU

    # ── model ─────────────────────────────────────────────────────────────────
    model_weights = args.model or config.get("model", "yolov8n.pt")
    print(f"[train_detector] Loading model: {model_weights}")
    model = YOLO(model_weights)

    # ── training args ─────────────────────────────────────────────────────────
    train_kwargs: dict = {
        "data": str(args.data_yaml),
        "device": device,
        "seed": args.seed,
        "exist_ok": args.exist_ok or config.get("exist_ok", False),
    }

    # CLI overrides take precedence over config file
    train_kwargs["epochs"]   = args.epochs   if args.epochs   is not None else config.get("epochs", 50)
    train_kwargs["patience"]  = args.patience if args.patience  is not None else config.get("patience", 10)
    train_kwargs["batch"]     = args.batch    if args.batch     is not None else config.get("batch", 16)
    train_kwargs["imgsz"]     = args.imgsz    if args.imgsz     is not None else config.get("imgsz", 640)
    train_kwargs["project"]   = str(args.project or config.get("project", "experiments/detector"))
    train_kwargs["name"]      = args.name     or config.get("name", "yolov8n_finetune")

    # Augmentation (from config, forwarded as-is by ultralytics)
    aug_keys = [
        "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear",
        "perspective", "flipud", "fliplr",
        "mosaic", "mixup", "copy_paste",
    ]
    for k in aug_keys:
        if k in config:
            train_kwargs[k] = config[k]

    # Learning schedule
    sched_keys = [
        "optimizer", "lr0", "lrf",
        "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    ]
    for k in sched_keys:
        if k in config:
            train_kwargs[k] = config[k]

    # Loss weights
    for k in ("box", "cls", "dfl"):
        if k in config:
            train_kwargs[k] = config[k]

    # Validation / output
    for k in ("val", "plots", "save", "save_json", "save_hybrid", "conf", "iou",
              "max_det", "workers", "amp", "pretrained", "verbose", "deterministic",
              "save_period"):
        if k in config:
            train_kwargs[k] = config[k]

    # CPU worker cap
    if "workers" not in train_kwargs:
        train_kwargs["workers"] = _cpu_count()

    # Mixed precision off for CPU
    if device == "cpu" and train_kwargs.get("amp", True):
        train_kwargs["amp"] = False

    # Resume
    train_kwargs["resume"] = args.resume

    print("\n[train_detector] Training arguments:")
    for k, v in sorted(train_kwargs.items()):
        print(f"  {k}: {v}")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[train_detector] Starting training on device='{device}' …")
    results = model.train(**train_kwargs)

    elapsed = time.time() - t0
    print(f"\n[train_detector] Training finished in {elapsed / 60:.1f} min")

    # ── best weights path ─────────────────────────────────────────────────────
    best_weights = Path(results.save_dir) / "weights/best.pt"
    if best_weights.exists():
        print(f"[train_detector] Best weights: {best_weights}")
    else:
        last_weights = Path(results.save_dir) / "weights/last.pt"
        print(f"[train_detector] Last weights: {last_weights}")

    # ── validation metrics ────────────────────────────────────────────────────
    val_metrics = results.results_dict
    if val_metrics:
        print("\n[train_detector] Validation metrics:")
        for k, v in val_metrics.items():
            if k.startswith("metrics/"):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("[train_detector] Done.")


if __name__ == "__main__":
    main()
