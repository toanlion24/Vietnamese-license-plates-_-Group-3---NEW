from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector training entrypoint placeholder.")
    parser.add_argument("--config", type=Path, default=Path("configs/detector/yolov8n.yaml"))
    parser.add_argument("--data-yaml", type=Path, default=Path("data/data.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    message = (
        "Skeleton only. Replace this script with YOLO/DeepSolo training call.\n"
        f"config={args.config}\n"
        f"data_yaml={args.data_yaml}\n"
        f"out_dir={args.out_dir}\n"
    )
    print(message)


if __name__ == "__main__":
    main()

