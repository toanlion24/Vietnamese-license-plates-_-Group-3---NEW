"""Chia manifest Buổi 4 (image_id, image_path, gt, ambiguous_gt) thành train / val."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.2, help="Tỉ lệ val (0–1).")
    p.add_argument("--out-train", type=Path, default=Path("data/manifests/train_manifest.csv"))
    p.add_argument("--out-val", type=Path, default=Path("data/manifests/val_manifest.csv"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = _read_rows(args.manifest)
    if not rows:
        raise SystemExit("Manifest rỗng.")

    fn = list(rows[0].keys())
    for req in ("image_id", "image_path", "gt"):
        if req not in fn:
            raise SystemExit(f"Thiếu cột bắt buộc {req} trong manifest.")

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 1 else 0
    if len(rows) == 1:
        n_val = 0
    val_set = set(idx[:n_val])
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_set]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_set]

    _write_rows(args.out_train, train_rows, fn)
    _write_rows(args.out_val, val_rows, fn)
    print(f"Train: {len(train_rows)} → {args.out_train}")
    print(f"Val:   {len(val_rows)} → {args.out_val}")


if __name__ == "__main__":
    sys.exit(main())
