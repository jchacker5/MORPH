"""Local, CPU-only experiment utilities. Invoke modules from the project root."""

import csv
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.stats import t

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mechanisms"
SEEDS = (11, 29, 47)
N_TEST = 2048


def tensor(a):
    return mx.array(a.astype(np.float32, copy=False))


def counts_one_hot(x, n):
    return np.eye(n, dtype=np.float32)[x]


def train(params, loss, batches, rate):
    """Small full-precision Adam optimizer with explicit evaluation each step."""
    grad_fn = mx.value_and_grad(loss)
    first = {k: mx.zeros_like(v) for k, v in params.items()}
    second = {k: mx.zeros_like(v) for k, v in params.items()}
    start = time.perf_counter()
    for step, batch in enumerate(batches, 1):
        value, grads = grad_fn(params, *batch)
        for k, grad in grads.items():
            first[k] = 0.9 * first[k] + 0.1 * grad
            second[k] = 0.999 * second[k] + 0.001 * grad * grad
            params[k] = params[k] - rate * (first[k] / (1 - 0.9**step)) / (
                mx.sqrt(second[k] / (1 - 0.999**step)) + 1e-8
            )
        mx.eval(params, first, second, value)
    return float(value.item()), time.perf_counter() - start


def verify_registration():
    actual = hashlib.sha256(
        (ROOT / "docs/mechanisms_protocol.md").read_bytes()
    ).hexdigest()
    expected = (OUT / "protocol.sha256").read_text().split()[0]
    if actual != expected:
        raise ValueError("Protocol hash changed")
    return actual


def save(name, rows):
    digest = verify_registration()
    ordered = [{**row, "protocol_sha256": digest} for row in rows]
    path = OUT / (name + ".json")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ordered, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    columns = list(dict.fromkeys(k for r in ordered for k in r))
    with path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in ordered:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in r.items()
                }
            )


def interval(x):
    a = np.asarray(x, dtype=float)
    d = float(t.ppf(0.975, len(a) - 1) * np.std(a, ddof=1) / np.sqrt(len(a)))
    return {
        "per_seed": a.tolist(),
        "mean": float(a.mean()),
        "low": float(a.mean() - d),
        "high": float(a.mean() + d),
    }
