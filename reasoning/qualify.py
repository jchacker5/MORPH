"""Train the two frozen checkpoint-qualification entries and retain both weights."""
from pathlib import Path
import subprocess
import sys
from common import OUT

for mode in ("adaptive","fixed2"):
    result = OUT / f"usable-w64-{mode}-s17.json"
    if result.exists():
        continue
    subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")),
                    "--mode",mode,"--seed","17","--width","64","--steps","8000",
                    "--optimized","--qualify"],check=True)
