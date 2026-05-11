from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
from collections.abc import Sequence

from src.postprocess.plate_rules import normalize_plate_text
from src.utils.types import EvalRecord


def _levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def levenshtein_chars(a: str, b: str) -> int:
    """Character edit distance on *normalized* plate strings."""
    ca = normalize_plate_text(a)
    cb = normalize_plate_text(b)
    return _levenshtein(list(ca), list(cb))


def cer(records: list[EvalRecord]) -> float:
    total_dist = 0
    total_chars = 0
    for r in records:
        gt = normalize_plate_text(r.gt)
        pred = normalize_plate_text(r.pred)
        total_dist += _levenshtein(list(gt), list(pred))
        total_chars += max(1, len(gt))
    return total_dist / total_chars if total_chars else 0.0


def _word_tokens(text: str) -> list[str]:
    tokens = [normalize_plate_text(part) for part in text.replace("\n", " ").split()]
    tokens = [token for token in tokens if token]
    if tokens:
        return tokens
    normalized = normalize_plate_text(text)
    return [normalized] if normalized else []


def wer(records: list[EvalRecord]) -> float:
    total_dist = 0
    total_words = 0
    for r in records:
        gt_words = _word_tokens(r.gt)
        pred_words = _word_tokens(r.pred)
        total_dist += _levenshtein(gt_words, pred_words)
        total_words += max(1, len(gt_words))
    return total_dist / total_words if total_words else 0.0


def plate_accuracy(records: list[EvalRecord]) -> float:
    if not records:
        return 0.0
    correct = sum(normalize_plate_text(r.gt) == normalize_plate_text(r.pred) for r in records)
    return correct / len(records)


def export_records(records: list[EvalRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "gt", "pred", "error_type"])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

