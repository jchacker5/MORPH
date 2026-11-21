"""Four trainable experts and four routing policies on controlled task families."""

import mlx.core as mx
import numpy as np

from .common import (
    N_TEST,
    SEEDS,
    counts_one_hot,
    save,
    tensor,
    train,
    verify_registration,
)

INPUTS, HIDDEN, EXPERTS, CLASSES = 18, 24, 4, 4


def data(rng, n):
    task = rng.permutation(np.arange(n) % 4)
    marker = task * 2 + rng.integers(2, size=n)
    a, b, c = rng.integers(4, size=n), rng.integers(4, size=n), rng.integers(2, size=n)
    x = np.concatenate(
        [
            counts_one_hot(marker, 8),
            counts_one_hot(a, 4),
            counts_one_hot(b, 4),
            counts_one_hot(c, 2),
        ],
        axis=1,
    )
    alternatives = np.stack(
        [(a + b) % 4, np.where(c, a, b), a, np.array([2, 0, 3, 1])[b]], axis=1
    )
    return x, alternatives[np.arange(n), task], task


def fit(mode, seed):
    rng = np.random.default_rng(seed)
    params = {
        "w1": tensor(rng.normal(scale=0.25, size=(4, INPUTS, HIDDEN))),
        "b1": tensor(np.zeros((4, 1, HIDDEN))),
        "w2": tensor(rng.normal(scale=0.2, size=(4, HIDDEN, 4))),
        "b2": tensor(np.zeros((4, 1, 4))),
    }
    has_gate = mode in ("risk", "switch")
    if has_gate:
        params.update(gw=tensor(0.15 * rng.normal(size=(INPUTS, 4))), gb=mx.zeros(4))

    def loss(p, x, target, random_route):
        all_logits = (
            mx.maximum(x[None] @ p["w1"] + p["b1"], 0) @ p["w2"] + p["b2"]
        ).transpose(1, 0, 2)

        def ce(logits, y):
            return mx.logsumexp(logits, axis=-1) - mx.sum(logits * y, axis=-1)

        if mode == "dense":
            return mx.mean(ce(mx.mean(all_logits, axis=1), target))
        if mode == "random":
            return mx.mean(
                ce(mx.sum(all_logits * random_route[:, :, None], axis=1), target)
            )
        prob = mx.softmax(x @ p["gw"] + p["gb"], axis=-1)
        usage = mx.mean(prob, axis=0)
        if mode == "risk":
            return mx.mean(
                mx.sum(ce(all_logits, target[:, None]) * prob, axis=-1)
            ) + 0.05 * 4 * mx.sum(usage**2)
        hard = mx.eye(4)[mx.stop_gradient(mx.argmax(prob, axis=-1))]
        chosen = mx.sum(all_logits * (prob * hard)[:, :, None], axis=1)
        return mx.mean(ce(chosen, target)) + 0.05 * 4 * mx.sum(
            mx.mean(hard, axis=0) * usage
        )

    stream = np.random.default_rng(seed + 3000)
    route_stream = np.random.default_rng(seed + 6000)

    def batches():
        for _ in range(500):
            x, y, _ = data(stream, 96)
            yield (
                tensor(x),
                tensor(counts_one_hot(y, 4)),
                tensor(counts_one_hot(route_stream.integers(4, size=len(y)), 4)),
            )

    last, seconds = train(params, loss, batches(), 0.012)
    p = {k: np.array(v) for k, v in params.items()}
    x, y, task = data(np.random.default_rng(seed + 4000), N_TEST)

    def expert(x, e):
        return np.maximum(x @ p["w1"][e] + p["b1"][e], 0) @ p["w2"][e] + p["b2"][e]

    if mode == "dense":
        pred = np.mean(np.stack([expert(x, e) for e in range(4)]), axis=0).argmax(-1)
        matrix = None
        mi = None
        forced_acc = None
    else:
        if has_gate:
            route = (x @ p["gw"] + p["gb"]).argmax(-1)
        else:
            route = np.random.default_rng(seed + 7000).integers(4, size=len(y))

        def dispatch(routes):
            scores = np.empty((len(y), 4), dtype=np.float32)
            for e in range(4):
                ids = np.flatnonzero(routes == e)
                scores[ids] = expert(x[ids], e)
            return scores.argmax(-1)

        pred = dispatch(route)
        forced_acc = float(np.mean(dispatch((route + 1) % 4) == y))
        freq = np.zeros((4, 4), dtype=int)
        np.add.at(freq, (task, route), 1)
        joint = freq / len(y)
        mask = joint > 0
        base = joint.sum(0, keepdims=True) * joint.sum(1, keepdims=True)
        mi = float(np.sum(joint[mask] * np.log2(joint[mask] / base[mask])))
        matrix = (freq / freq.sum(1, keepdims=True)).tolist()
    expert_params = sum(p[k].size for k in ("w1", "b1", "w2", "b2")) // 4
    gate_params = INPUTS * 4 + 4 if has_gate else 0
    return {
        "seed": seed,
        "mode": mode,
        "accuracy": float(np.mean(pred == y)),
        "task_accuracy": [
            float(np.mean(pred[task == i] == y[task == i])) for i in range(4)
        ],
        "mi_bits": mi,
        "task_expert_matrix": matrix,
        "forced_accuracy": forced_acc,
        "params": expert_params * 4 + gate_params,
        "active_params": expert_params * (4 if mode == "dense" else 1) + gate_params,
        "train_experts_materialized": 4,
        "train_examples": 500 * 96,
        "test_examples": N_TEST,
        "final_train_objective": last,
        "train_seconds": seconds,
    }


if __name__ == "__main__":
    verify_registration()
    rows = []
    for seed in SEEDS:
        for mode in ("risk", "switch", "random", "dense"):
            rows.append(fit(mode, seed))
        save("routing", rows)
        print("routing seed completed", seed, flush=True)
