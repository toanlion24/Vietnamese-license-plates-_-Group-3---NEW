from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def strip_notebook(path: Path) -> tuple[int, bool]:
    with path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    changed = False
    cells = notebook.get("cells", [])
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if changed:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
            f.write("\n")

    return len(cells), changed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove outputs and execution_count from .ipynb files.",
    )
    parser.add_argument("notebooks", nargs="+", help="Notebook path(s) to clean.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    candidate_paths: list[Path] = []
    for notebook_arg in args.notebooks:
        if any(ch in notebook_arg for ch in "*?[]"):
            matched = [Path(p) for p in glob.glob(notebook_arg, recursive=True)]
            if not matched:
                print(f"[skip] No file matched pattern: {notebook_arg}")
            candidate_paths.extend(matched)
        else:
            candidate_paths.append(Path(notebook_arg))

    for notebook_path in candidate_paths:
        if not notebook_path.exists():
            print(f"[skip] Not found: {notebook_path}")
            continue
        if notebook_path.suffix.lower() != ".ipynb":
            print(f"[skip] Not a notebook: {notebook_path}")
            continue

        cells_count, changed = strip_notebook(notebook_path)
        status = "cleaned" if changed else "already_clean"
        print(f"[{status}] {notebook_path} (cells={cells_count})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
