"""Answer-supervised memory with an actual bounded streaming inference path."""

import time

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

CONDITIONS = ("iid", "cue_shift", "recency_shift", "both", "both_long")
MODES = ("full", "window", "stride", "learned", "permuted", "cue_rule")


def data(rng, n, length, condition):
    shifted = condition in ("cue_shift", "both", "both_long")
    early = condition in ("recency_shift", "both", "both_long")
    cue = rng.choice(4, size=n, p=[0, 0, 0.5, 0.5] if shifted else [0.4, 0.4, 0.1, 0.1])
    relevant = rng.random((n, length // 4 if early else length)).argsort(axis=1)[:, :4]
    colors = (cue[:, None] + rng.integers(1, 4, size=(n, length))) % 4
    colors[np.arange(n)[:, None], relevant] = cue[:, None]
    keys = np.argsort(rng.random((n, length)), axis=1).astype(np.uint16)
    values = rng.integers(16, size=(n, length), dtype=np.uint8)
    qpos = relevant[np.arange(n), rng.integers(4, size=n)]
    return cue, colors, keys, values, qpos


def features(cue, colors):
    b, l = colors.shape
    return np.concatenate(
        (
            np.broadcast_to(counts_one_hot(cue, 4)[:, None, :], (b, l, 4)),
            counts_one_hot(colors, 4),
            counts_one_hot(cue[:, None] * 4 + colors, 16),
            np.broadcast_to(
                (np.arange(l, dtype=np.float32) / max(l - 1, 1))[None, :, None],
                (b, l, 1),
            ),
        ),
        axis=2,
    )


def fit(seed, k):
    rng = np.random.default_rng(seed)
    params = {
        "w": tensor(rng.normal(scale=0.2, size=(25, 12))),
        "b": mx.zeros(12),
        "out": tensor(rng.normal(scale=0.2, size=(12, 1))),
        "bias": mx.zeros(1),
    }

    def loss(p, x, hit):
        score = (mx.tanh(x @ p["w"] + p["b"]) @ p["out"] + p["bias"])[:, :, 0]
        # Constrain the sum of sigmoid inclusion probabilities to k via bisection.
        # Implicit derivative d cutoff / d score_i = sigmoid'_i / sum sigmoid'.
        # This is the gradient of the same soft capacity equation.
        s = mx.stop_gradient(score)
        lo = mx.min(s, axis=-1, keepdims=True) - 20
        hi = mx.max(s, axis=-1, keepdims=True) + 20
        for _ in range(16):
            mid = (hi + lo) * 0.5
            high_mass = mx.sum(mx.sigmoid(s - mid), axis=-1, keepdims=True) > k
            lo = mx.where(high_mass, mid, lo)
            hi = mx.where(high_mass, hi, mid)
        threshold = (hi + lo) * 0.5
        keep0 = mx.sigmoid(s - threshold)
        slopes = keep0 * (1 - keep0)
        weights = mx.stop_gradient(
            slopes / mx.maximum(mx.sum(slopes, axis=1, keepdims=True), 1e-8)
        )
        cutoff = threshold + mx.sum(weights * (score - s), axis=1, keepdims=True)
        keep = mx.sigmoid(score - cutoff)
        # Single matching key. Correct retained value has unit probability,
        # missing key receives a uniform predictive distribution over 16 values.
        return -mx.mean(mx.log(1 / 16 + (15 / 16) * mx.sum(keep * hit, axis=1)))

    stream = np.random.default_rng(seed + 8000)

    def batches():
        for _ in range(250):
            cue, color, keys, values, qpos = data(stream, 64, 32, "iid")
            ar = np.arange(len(cue))
            query = keys[ar, qpos]
            target = values[ar, qpos]
            matched = (keys == query[:, None]) & (values == target[:, None])
            yield tensor(features(cue, color)), tensor(matched)

    _, seconds = train(params, loss, batches(), 0.02)
    return {a: np.array(v) for a, v in params.items()}, seconds


def raw_priority(p, cue, colors):
    # No temporal aggregation: a record's score depends only on cue/color/offset.
    return (np.tanh(features(cue, colors) @ p["w"] + p["b"]) @ p["out"] + p["bias"])[
        :, :, 0
    ]


def chosen(mode, p, cue, colors, k, shuffle):
    b, length = colors.shape
    ar = np.arange(length)
    if mode == "full":
        return np.broadcast_to(ar, (b, length))
    if mode == "window":
        return np.broadcast_to(ar[-k:], (b, k))
    if mode == "stride":
        return np.broadcast_to(ar[(ar * k) % length == 0], (b, k))
    if mode == "cue_rule":
        return np.where(colors == cue[:, None])[1].reshape(b, 4)
    scores = raw_priority(p, cue, colors)
    if mode == "permuted":
        scores = np.take_along_axis(scores, shuffle, axis=1)
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def reference_stream(p, mode, cue, colors, keys, values, qkey, k, shuffle=None):
    # Bounded allocated cache. Input and learned constants are external.
    length = len(colors)
    size = length if mode == "full" else k
    keybuf = np.zeros(size, dtype=np.uint16)
    valbuf = np.zeros(size, dtype=np.uint8)
    scorebuf = (
        np.empty(size, dtype=np.float32) if mode in ("learned", "permuted") else None
    )
    nvalid = 0
    if scorebuf is not None:
        # Precompute affine offsets by color once per episode; position is scalar.
        w = p["w"]
        base = w[cue] + w[4:8] + w[8 + 4 * cue : 12 + 4 * cue] + p["b"]

        def score(i):
            return float(
                (np.tanh(base[colors[i]] + w[24] * (i / (length - 1))) @ p["out"][:, 0])
                + p["bias"][0]
            )

    for i in range(length):
        if mode == "stride" and (i * k) % length:
            continue
        if mode == "cue_rule" and (colors[i] != cue or nvalid == k):
            continue
        if scorebuf is not None:
            priority = score(shuffle[i] if shuffle is not None else i)
            if nvalid == size:
                j = int(np.argmin(scorebuf))
                if priority <= scorebuf[j]:
                    continue
            else:
                j = nvalid
            scorebuf[j] = priority
        else:
            j = i % k if mode == "window" else nvalid
        keybuf[j], valbuf[j] = keys[i], values[i]
        nvalid = min(size, nvalid + 1)
    matches = np.flatnonzero(keybuf[:nvalid] == qkey)
    return (
        int(valbuf[matches[0]]) if len(matches) else 0,
        bool(len(matches)),
        keybuf.nbytes
        + valbuf.nbytes
        + (scorebuf.nbytes if scorebuf is not None else 0),
    )


def evaluate(seed, k, p, seconds, cond):
    length = 96 if cond == "both_long" else 32
    stream = np.random.default_rng(seed + 9000 + CONDITIONS.index(cond))
    cue, colors, keys, values, qpos = data(stream, N_TEST, length, cond)
    shuffle = np.argsort(stream.random((N_TEST, length)), axis=1)
    ar = np.arange(N_TEST)
    actual = values[ar, qpos]
    rows = []
    for mode in MODES:
        keep = chosen(mode, p, cue, colors, k, shuffle)
        hit = np.any(keep == qpos[:, None], axis=-1)
        pred = np.where(hit, actual, 0)
        times = []
        # Vectorized scoring and scalar reference path must agree on timing subset.
        scan_n = 48
        for _ in range(3):
            start = time.perf_counter_ns()
            measured = [
                reference_stream(
                    p,
                    mode,
                    cue[i],
                    colors[i],
                    keys[i],
                    values[i],
                    keys[i, qpos[i]],
                    k,
                    shuffle[i] if mode == "permuted" else None,
                )
                for i in range(scan_n)
            ]
            times.append((time.perf_counter_ns() - start) / 1e3 / scan_n)
            assert [x[0] for x in measured] == pred[:scan_n].tolist()
            assert [x[1] for x in measured] == hit[:scan_n].tolist()
        rows.append(
            {
                "seed": seed,
                "slots": k,
                "condition": cond,
                "length": length,
                "mode": mode,
                "accuracy": float(np.mean(pred == actual)),
                "key_hit": float(hit.mean()),
                "cache_bytes": measured[0][2],
                "latency_us": float(np.median(times)),
                "latency_blocks_us": times,
                "timing_examples_per_block": scan_n,
                "test_examples": N_TEST,
                "params": sum(v.size for v in p.values())
                if mode in ("learned", "permuted")
                else 0,
                "train_examples": 250 * 64 if mode in ("learned", "permuted") else 0,
                "train_seconds": seconds if mode in ("learned", "permuted") else 0.0,
            }
        )
    return rows


if __name__ == "__main__":
    verify_registration()
    rows = []
    for seed in SEEDS:
        for k in (4, 8):
            p, seconds = fit(seed, k)
            for c in CONDITIONS:
                rows.extend(evaluate(seed, k, p, seconds, c))
        save("memory", rows)
        print("memory seed completed", seed, flush=True)
