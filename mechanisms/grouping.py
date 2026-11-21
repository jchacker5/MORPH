"""Causal keyed-fact episodes with learned compressed memory."""

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

FAMILIES = np.array([2, 0, 1, 3, 0, 3, 2, 1, 3, 1, 2, 0])
LENGTH = 16
# Serialized prefix: symbol, value+12 pairs, then query symbol, target+12, suffix.
QUERY_POS = LENGTH * 2


def generate(rng, batch):
    keys = rng.integers(12, size=(batch, LENGTH))
    query = rng.integers(12, size=batch)
    family_values = rng.integers(4, size=(batch, 4))
    values = family_values[np.arange(batch)[:, None], FAMILIES[keys]]
    labels = family_values[np.arange(batch), FAMILIES[query]]
    seq = np.empty((batch, QUERY_POS + 4), dtype=np.int32)
    seq[:, :QUERY_POS:2] = keys
    seq[:, 1:QUERY_POS:2] = values + 12
    seq[:, QUERY_POS] = query
    seq[:, QUERY_POS + 1] = labels + 12
    seq[:, QUERY_POS + 2 :] = rng.integers(16, size=(batch, 2))
    return seq


def decode(seq):
    seen = seq[:, : QUERY_POS + 1]
    return (
        mx.array(seen[:, :QUERY_POS:2]),
        tensor(counts_one_hot(seen[:, 1:QUERY_POS:2] - 12, 4)),
        mx.array(seen[:, QUERY_POS]),
    )


def adjusted_rand(x, y):
    c = np.zeros((x.max() + 1, y.max() + 1), dtype=np.int64)
    np.add.at(c, (x, y), 1)
    pairs = lambda v: (v * (v - 1) // 2).sum()
    a, b = pairs(c.sum(0)), pairs(c.sum(1))
    expected = float(a) * float(b) / pairs(np.array([len(x)]))
    return float((pairs(c) - expected) / ((a + b) / 2 - expected))


def fit(mode, k, seed):
    rng = np.random.default_rng(seed + 90)
    params = {"affinity": tensor(0.08 * rng.normal(size=(12, 12)))}
    if mode == "learned":
        params["write"] = tensor(0.12 * rng.normal(size=(12, k)))
    random_map = rng.permutation(12) % k
    static_map = {"random": random_map, "oracle": FAMILIES}.get(mode)
    positional = {
        "adjacent": np.arange(LENGTH) * k // LENGTH,
        "stride": np.arange(LENGTH) % k,
        "uncompressed": np.arange(LENGTH),
    }.get(mode)
    static = (
        tensor(counts_one_hot(static_map if static_map is not None else positional, k))
        if mode != "learned"
        else None
    )

    def forward(p, keys, values, query, hard=False):
        if mode == "learned":
            w = mx.softmax(p["write"], axis=-1)
            if hard:
                w = mx.eye(k)[mx.argmax(w, axis=-1)]
            writes = w[keys]
        elif static_map is not None:
            writes = static[keys]
        else:
            writes = mx.broadcast_to(static[None], (keys.shape[0], LENGTH, k))
        # Sufficient statistics of observed keys and values within each slot.
        memory = writes.transpose(0, 2, 1) @ values
        address = writes.transpose(0, 2, 1) @ mx.eye(12)[keys]
        query_affinity = mx.softmax(p["affinity"][query], axis=-1)
        slot_mass = (address @ query_affinity[:, :, None])[:, :, 0]
        read = slot_mass / mx.maximum(mx.sum(slot_mass, axis=1, keepdims=True), 1e-12)
        value_frequency = (memory + 0.01) / (
            mx.sum(memory, axis=-1, keepdims=True) + 0.04
        )
        return mx.sum(read[:, :, None] * value_frequency, axis=1), writes

    def loss(p, keys, values, query, target):
        prob, _ = forward(p, keys, values, query)
        return -mx.mean(mx.log(mx.sum(prob * target, axis=-1)))

    stream = np.random.default_rng(seed + 1000)

    def batches():
        for _ in range(300):
            seq = generate(stream, 64)
            yield (*decode(seq), tensor(counts_one_hot(seq[:, QUERY_POS + 1] - 12, 4)))

    _, seconds = train(params, loss, batches(), 0.04)
    test = generate(np.random.default_rng(seed + 2000), N_TEST)
    pred, w = forward(params, *decode(test), hard=True)
    prob, slots = np.array(pred), np.array(mx.argmax(w, axis=-1))
    mutated = test.copy()
    mutated[:, QUERY_POS + 1 :] = np.random.default_rng(seed + 5000).integers(
        16, size=mutated[:, QUERY_POS + 1 :].shape
    )
    changed, _ = forward(params, *decode(mutated), hard=True)
    max_change = float(np.max(np.abs(prob - np.array(changed))))
    assert max_change == 0
    labels = test[:, QUERY_POS + 1] - 12
    nll = float(-np.log(prob[np.arange(N_TEST), labels]).mean())
    families = FAMILIES[test[:, :QUERY_POS:2]]
    return {
        "seed": seed,
        "mode": mode,
        "slots": k,
        "nominal_compression": LENGTH / k,
        "slot_histogram_bytes": k * 16 * 4,
        "params": sum(v.size for v in params.values()),
        "accuracy": float(np.mean(prob.argmax(-1) == labels)),
        "logloss": nll,
        "ppl": float(np.exp(nll)),
        "ari": adjusted_rand(families.reshape(-1), slots.reshape(-1)),
        "max_suffix_delta": max_change,
        "train_examples": 300 * 64,
        "test_examples": N_TEST,
        "train_seconds": seconds,
    }


if __name__ == "__main__":
    verify_registration()
    rows = []
    for seed in SEEDS:
        for k in (2, 4, 8):
            for mode in ("learned", "random", "adjacent", "stride"):
                rows.append(fit(mode, k, seed))
        rows.extend([fit("uncompressed", 16, seed), fit("oracle", 4, seed)])
        save("grouping", rows)
        print("grouping seed completed", seed, flush=True)
