import subprocess
import sys
import os

# Find the exact file
scripts_dir = os.path.join("D:", os.sep, "ComputerVisionNew", "scripts")
files = os.listdir(scripts_dir)
eval_files = [f for f in files if 'eval' in f.lower() and 'tro' in f.lower()]

print(f"Found eval files: {eval_files}")

if eval_files:
    # Use the first match
    script_path = os.path.join(scripts_dir, eval_files[0])
    print(f"Using script: {script_path}")
    print(f"File exists: {os.path.exists(script_path)}")

    # Read first 200 chars
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"File content (first 200 chars):\n{content[:200]}")

    # Run the script
    cmd = [
        sys.executable,
        script_path,
        "--manifest", "D:/ComputerVisionNew/data/manifests/real_plates_manifest_39.csv",
        "--configs", "A,B,C,D"
    ]

    print(f"\nRunning command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="D:/ComputerVisionNew")
    sys.exit(result.returncode)
else:
    print("No eval_tro*_configs.py found!")
