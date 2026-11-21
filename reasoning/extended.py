"""Launch all 45 predeclared longer scaling fits, with resumable outputs."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

from common import OUT, SEEDS

for width, seed, mode in itertools.product((896,1600,2880), SEEDS, ("fixed1","fixed2","fixed3","fixed4","adaptive")):
    path = OUT / f"long-w{width}-{mode}-s{seed}.json"
    if path.exists():
        assert json.loads(path.read_text())["updates"] == 160
        continue
    subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")), "--mode", mode,
                    "--width",str(width),"--seed",str(seed),"--steps","160","--scale","--extended"], check=True)
