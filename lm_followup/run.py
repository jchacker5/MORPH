"""Budget-balanced replay, longer paired comparison and a five-minute capstone."""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lm"))
from generate import generate
from prepare import PREFIX, WINDOW, evaluation_windows, split_data
from train import Decoder, fit, score

from lm_followup.data import DIGEST, eval_windows, poems

OUT = ROOT / "results/lm_followup"
FROZEN = ROOT / "results/lm/frozen.json"
SEEDS = (11, 23, 37)
PROMPT = "To be, or not to be"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, row):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def registration():
    actual = digest(ROOT / "docs/lm_followup_protocol.md")
    assert actual == (OUT / "protocol.sha256").read_text().split()[0]
    for name, d in json.loads(FROZEN.read_text())["source"].items():
        assert digest(ROOT / "lm" / name) == d, f"Frozen training code changed: {name}"
    return actual


def saved_model(model, weights, examples):
    model.save_weights(str(weights))
    fresh = Decoder(True)
    fresh.load_weights(str(weights))
    sample = mx.array(examples[:8, :-1])
    expected, depth = model.infer(sample, "adaptive")
    read, depth2 = fresh.infer(sample, "adaptive")
    mx.eval(read, expected)
    difference = float(mx.max(mx.abs(read - expected)).item())
    assert difference == 0 and np.array_equal(depth, depth2)
    first = generate(weights, PROMPT, 160, 0.8, 73)
    second = generate(weights, PROMPT, 160, 0.8, 73)
    assert first == second and len(first) == len(PROMPT.encode()) + 160
    return {
        "weights_file": weights.name,
        "sha256": digest(weights),
        "max_abs_logit_reload_delta": difference,
        "equal_reloaded_depths": True,
        "sample_hex": first.hex(),
        "sample_utf8": first.decode("utf8", errors="replace"),
        "generation_repeat_identical": True,
    }


def latency(model, mode, examples):
    for _ in range(4):
        mx.eval(model.infer(mx.array(examples[:1, :-1]), mode)[0])
    elapsed = []
    for row in examples[:40]:
        x = mx.array(row[None, :-1])
        start = time.perf_counter_ns()
        mx.eval(model.infer(x, mode)[0])
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"batch1_median_ms": float(np.median(elapsed)), "latency_batch1_ms": elapsed}


def run_cell(stage, seed, mode, trial=None):
    cfg = json.loads(FROZEN.read_text())["config"]
    if stage == "replay":
        plan = ROOT / f"results/lm/search/trial-{trial:02d}.plan.json"
        cfg = json.loads(plan.read_text())["config"]
        path = OUT / f"replay-{trial}.json"
        budget = {"steps": 2048}
    elif stage == "capstone":
        path = OUT / "capstone.json"
        budget = {"seconds": 300}
    else:
        path = OUT / f"long-{mode}-{seed}.json"
        budget = {"steps": 4096}
    protocol = registration()
    if path.exists():
        row = json.loads(path.read_text())
        assert (
            row["protocol_sha256"] == protocol
            and row["mode"] == mode
            and row["seed"] == seed
        )
        return
    model, row = fit(mode, seed, **cfg, **budget)
    row.update(protocol_sha256=protocol, stage=stage)
    if stage in ("capstone", "replay"):
        row["validation"] = score(model, mode, evaluation_windows("val"))
        row["validation_objective"] = row["validation"]["bpb"] + 0.08 * max(
            0.0, row["validation"]["mean_depth"] - 2
        )
    if stage != "replay":
        examples = eval_windows()
        row["sonnets"] = score(model, mode, examples)
        row["repeat_test"] = score(model, mode, evaluation_windows("test"))
        row.update(latency(model, mode, examples))
        if mode == "adaptive" and (stage == "capstone" or seed == 11):
            row["checkpoint_verification"] = saved_model(
                model, OUT / (path.stem + ".safetensors"), examples
            )
    write(path, row)
    print(
        "finished",
        path.name,
        "steps",
        row["steps"],
        "training_seconds",
        round(row["seconds"], 3),
        flush=True,
    )


def count_comparisons():
    train = split_data("train").tobytes()
    counts = [defaultdict(Counter) for _ in range(4)]
    for i, char in enumerate(train):
        for n in range(min(3, i) + 1):
            counts[n][train[i - n : i]][char] += 1
    totals = [{k: sum(c.values()) for k, c in group.items()} for group in counts]

    def evaluate(examples):
        bits = {1: 0.0, 3: 0.0}
        for row in examples:
            s = row.astype("uint8").tobytes()
            for i in range(PREFIX + 1, WINDOW + 1):
                ch = s[i]
                p = (counts[0][b""][ch] + 0.5) / (len(train) + 128)
                for n in (1, 2, 3):
                    key = s[i - n : i]
                    hist = counts[n].get(key)
                    if hist:
                        distinct = len(hist)
                        p = (hist[ch] + distinct * p) / (totals[n][key] + distinct)
                    if n in bits:
                        bits[n] -= np.log2(p)
        ntarget = len(examples) * (WINDOW - PREFIX)
        return {
            "bpb": {str(k): float(x / ntarget) for k, x in bits.items()},
            "examples": len(examples),
            "scored_bytes": ntarget,
        }

    external = eval_windows()
    row = {
        "protocol_sha256": registration(),
        "training_bytes": len(train),
        "source_sha256": DIGEST,
        "sonnets": evaluate(external),
        "repeat_test": evaluate(evaluation_windows("test")),
        "poem_count": len(poems()),
        "clean_bytes": sum(map(len, poems())),
        "cleaned_poems_delimited_by_nul_sha256": hashlib.sha256(
            b"\0".join(poems())
        ).hexdigest(),
        "exact_external_windows_present_in_training": 0,
    }
    original = json.loads((ROOT / "results/lm/counts.json").read_text())
    assert np.allclose(
        list(row["repeat_test"]["bpb"].values()),
        list(original["bpb"].values()),
        rtol=0,
        atol=1e-10,
    )
    write(OUT / "counts.json", row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("capstone", "replay", "long", "counts", "one"))
    p.add_argument("--seed", type=int)
    p.add_argument("--mode")
    p.add_argument("--trial", type=int)
    p.add_argument("--cell-stage", choices=("replay", "long"))
    args = p.parse_args()
    if args.stage == "counts":
        count_comparisons()
    elif args.stage == "capstone":
        run_cell("capstone", 7, "adaptive")
    elif args.stage == "one":
        run_cell(args.cell_stage, args.seed, args.mode, args.trial)
    else:
        # Release compiled graph caches and GPU buffers between independent fits.
        import subprocess

        for seed, mode, trial in (
            [(7, "adaptive", k) for k in (1, 2, 3, 4)]
            if args.stage == "replay"
            else [(s, m, 0) for s in SEEDS for m in ("1", "2", "4", "adaptive")]
        ):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lm_followup.run",
                    "one",
                    "--cell-stage",
                    args.stage,
                    "--seed",
                    str(seed),
                    "--mode",
                    mode,
                    "--trial",
                    str(trial),
                ],
                cwd=ROOT,
                check=True,
            )
