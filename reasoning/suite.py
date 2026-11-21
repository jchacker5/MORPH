"""Run the frozen matrix, resuming only atomic final JSON results."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import OUT, SEEDS

HERE = Path(__file__).resolve().parent


def run(mode, width, seed, steps, scale=False, profile=False):
    prefix = "profile" if profile else "scale" if scale else "main"
    path = OUT / f"{prefix}-w{width}-{mode}-s{seed}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["updates"] != steps:
            raise RuntimeError(f"Step mismatch at {path}. Repair or archive the previous run.")
        return saved
    command = [sys.executable, str(HERE / "run.py"), "--mode", mode,
               "--seed", str(seed), "--width", str(width), "--steps", str(steps)]
    if scale:
        command.append("--scale")
    if profile:
        command.append("--profile")
    subprocess.run(command, check=True)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["all", "main", "scale"], default="all", nargs="?")
    part = parser.parse_args().part
    if part in ("main","all"):
        for seed, mode in itertools.product(SEEDS, ("adaptive","fixed1","fixed2","fixed3","fixed4","untied4","random")):
            run(mode, 32 if mode == "untied4" else 64, seed, 120)
    if part in ("scale","all"):
        profile = run("adaptive",2880,9,3,True,True)
        seconds = np.median(profile["train_step_seconds"][1:])
        steps = 4 if seconds > 5 else 8 if seconds > 2 else 16
        print(f"Width-2880 warmed step {seconds:.3f} s. Frozen rule selected {steps} updates.", flush=True)
        for width, seed, mode in itertools.product((896,1600,2880), SEEDS, ("fixed4","adaptive")):
            run(mode,width,seed,steps,True)


if __name__ == "__main__":
    main()
