"""Entry point — runs _eval_real.py which sanitizes sys.path at its first line."""
import sys
from pathlib import Path

# Remove ALL ComputerVisionNew paths before running the real script
for p in list(sys.path):
    if "ComputerVisionNew" in p:
        sys.path.remove(p)

# Now exec the real script (it will re-sanitize on its own first line)
_real = Path(__file__).parent / "_eval_real.py"
with open(_real, encoding="utf-8") as f:
    exec(compile(f.read(), str(_real), "exec"))
