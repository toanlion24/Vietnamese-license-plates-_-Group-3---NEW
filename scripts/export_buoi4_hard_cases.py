"""Export top hard cases (max char error vs GT) for Buổi 4 Markdown appendix."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics_plate import levenshtein_chars
from src.postprocess.plate_rules import normalize_plate_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-a-csv", type=Path, required=True)
    p.add_argument("--config-b-csv", type=Path, required=True)
    p.add_argument("--output-md", type=Path, default=Path("reports/buoi4_hard_cases.md"))
    p.add_argument("--limit", type=int, default=20)
    return p.parse_args()


def _first_nonempty(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        v = (row.get(key) or "").strip()
        if v:
            return v
    return ""


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fields = reader.fieldnames or ()
        for row in reader:
            iid = _first_nonempty(row, ("image_id", "id", "filename"))
            if not iid:
                continue
            by_id[iid] = {k: (row.get(k) or "") for k in fields}
    return by_id


def main() -> None:
    args = parse_args()
    a_map = load_rows(args.config_a_csv)
    b_map = load_rows(args.config_b_csv)
    common = sorted(set(a_map) & set(b_map))
    if not common:
        raise SystemExit("Hai CSV không có image_id chung.")

    ranked: list[tuple[int, str, str, str, str, str, str]] = []
    for iid in common:
        ra, rb = a_map[iid], b_map[iid]
        gt = _first_nonempty(ra, ("gt", "text_gt", "label"))
        if not gt:
            gt = _first_nonempty(rb, ("gt", "text_gt", "label"))
        pa = _first_nonempty(ra, ("pred", "text_pred", "plate_text"))
        pb = _first_nonempty(rb, ("pred", "text_pred", "plate_text"))
        ea = _first_nonempty(ra, ("error_type",))
        eb = _first_nonempty(rb, ("error_type",))
        if normalize_plate_text(gt) == normalize_plate_text(pa) and normalize_plate_text(gt) == normalize_plate_text(pb):
            continue
        score = max(levenshtein_chars(gt, pa), levenshtein_chars(gt, pb))
        ranked.append((score, iid, gt, pa, pb, ea, eb))

    ranked.sort(key=lambda t: (-t[0], t[1]))
    take = ranked[: max(0, args.limit)]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Buổi 4 — hard cases (tự động)",
        "",
        f"- Nguồn A: `{args.config_a_csv}`",
        f"- Nguồn B: `{args.config_b_csv}`",
        f"- Top {len(take)} mẫu sai theo max(levenshtein(gt,pred_A), levenshtein(gt,pred_B)).",
        "",
        "| rank | image_id | GT | pred A | pred B | error A | error B | max dist |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, (score, iid, gt, pa, pb, ea, eb) in enumerate(take, start=1):
        lines.append(
            f"| {idx} | `{iid}` | `{gt}` | `{pa}` | `{pb}` | {ea or '—'} | {eb or '—'} | {score} |"
        )
    lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_md} ({len(take)} rows)")


if __name__ == "__main__":
    main()
