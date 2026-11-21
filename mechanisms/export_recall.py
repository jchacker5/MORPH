"""Deterministically replay the first registered memory seed and save its weights."""

import json

import numpy as np

from .common import OUT, SEEDS, verify_registration
from .memory import CONDITIONS, evaluate, fit

if __name__ == "__main__":
    verify_registration()
    seed = SEEDS[0]
    p, seconds = fit(seed, 4)
    expected = {
        (r["mode"], r["condition"]): r
        for r in json.loads((OUT / "memory.json").read_text())
        if r["seed"] == seed and r["slots"] == 4
    }
    for c in CONDITIONS:
        for row in evaluate(seed, 4, p, seconds, c):
            registered = expected[row["mode"], c]
            assert (
                registered["accuracy"] == row["accuracy"]
                and registered["key_hit"] == row["key_hit"]
            )
    np.savez(OUT / "recall-seed11-k4.npz", **p)
    print(
        "Exported first registered K=4 memory model, replay matched all five conditions."
    )
