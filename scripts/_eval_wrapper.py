#!/usr/bin/env python3
"""
Entry point — passes control to eval_lora_compare.py in a fresh subprocess.
Uses sys.executable to avoid any shell-level cached state.
"""
import subprocess, sys, os

# Build a clean env with sklearn/pandas/pyarrow blocked
clean_env = os.environ.copy()
clean_env["PYARROW_CSV_IPC_ENABLE"] = "0"
clean_env["ARROW_DISABLE_MMAP"] = "1"

# Point PYTHONPATH to project
project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clean_env["PYTHONPATH"] = project

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_lora_compare.py")

result = subprocess.run(
    [sys.executable, script, "--max-samples", "550"],
    env=clean_env,
    capture_output=False,
)
sys.exit(result.returncode)
