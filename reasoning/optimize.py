"""Complete the precommitted 2,400-update follow-up matrix."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

from common import OUT, SEEDS

for seed, mode in itertools.product(SEEDS, ("adaptive","fixed1","fixed2","fixed3","fixed4","untied4","random")):
    width = 32 if mode == "untied4" else 64
    path = OUT / f"opt-w{width}-{mode}-s{seed}.json"
    if path.exists():
        assert json.loads(path.read_text())["updates"] == 2400
        continue
    subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")), "--mode", mode,
                    "--width",str(width),"--seed",str(seed),"--steps","2400","--optimized"], check=True)
