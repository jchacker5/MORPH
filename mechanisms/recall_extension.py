"""Supplementary dense-attention and associative-cache comparisons."""

import hashlib
import time

import numpy as np

from .common import N_TEST, OUT, ROOT, SEEDS, save, verify_registration
from .memory import data, fit, raw_priority, reference_stream

BITS = np.arange(16, dtype=np.uint16)
POLICIES = ("full", "attention", "hash", "learned", "cue_rule")


def attention(keys, values, q):
    # Compute single-query dot-product attention on bipolar bits. Only expanded
    # transient Q and K tensors use float32; retain original packed keys.
    unpack = lambda a: 2 * ((a[..., None] >> BITS) & 1).astype(np.float32) - 1
    qvec = unpack(q)
    kvec = unpack(keys)
    scores = ((kvec @ qvec[:, :, None])[:, :, 0]) * 8
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    vote = (weights[:, None, :] @ np.eye(16, dtype=np.float32)[values])[:, 0]
    return vote.argmax(-1), np.take_along_axis(
        keys, weights.argmax(-1)[:, None], axis=1
    )[:, 0] == q


def direct_map(keys, values, qkey):
    kb = np.empty(4, dtype=np.uint16)
    vb = np.empty(4, dtype=np.uint8)
    for key, value in zip(keys, values):
        slot = int(key) % 4
        kb[slot], vb[slot] = key, value
    loc = int(qkey) % 4
    found = bool(kb[loc] == qkey)
    return int(vb[loc]) if found else 0, found, int(kb.nbytes + vb.nbytes)


def scan(mode, p, cue, col, key, val, q):
    if mode == "attention":
        # Allocation and prefix copy are charged in latency and cache size.
        kb = key.copy()
        vb = val.copy()
        predicted, hit = attention(kb[None], vb[None], np.array([q], dtype=np.uint16))
        return int(predicted[0]), bool(hit[0]), int(kb.nbytes + vb.nbytes)
    if mode == "hash":
        return direct_map(key, val, q)
    return reference_stream(p, mode, cue, col, key, val, q, 4)


def run(seed, length, p):
    stream = np.random.default_rng(51000 + seed + length)
    totals = {a: np.zeros(2, dtype=int) for a in POLICIES}
    latency = {}
    byte_counts = {}
    for offset in range(0, N_TEST, 32):
        cue, col, key, val, pos = data(stream, 32, length, "both")
        ix = np.arange(len(cue))
        query = key[ix, pos]
        truth = val[ix, pos]
        for mode in POLICIES:
            if mode == "attention":
                pred, hit = attention(key, val, query)
            else:
                if mode == "learned":
                    slots = np.argsort(
                        -raw_priority(p, cue, col), axis=1, kind="stable"
                    )[:, :4]
                    hit = (slots == pos[:, None]).any(1)
                elif mode == "hash":
                    candidate = (key % 4) == (query % 4)[:, None]
                    final_pos = length - 1 - candidate[:, ::-1].argmax(1)
                    hit = final_pos == pos
                elif mode == "cue_rule":
                    slots = np.nonzero(col == cue[:, None])[1].reshape(len(cue), 4)
                    hit = (slots == pos[:, None]).any(1)
                else:
                    hit = (key == query[:, None]).any(1)
                pred = np.where(hit, truth, 0)
            totals[mode] += np.array([(pred == truth).sum(), hit.sum()])
            if offset:
                continue
            blocks = []
            for _ in range(3):
                start = time.perf_counter_ns()
                actual = [
                    scan(mode, p, cue[i], col[i], key[i], val[i], query[i])
                    for i in range(16)
                ]
                blocks.append((time.perf_counter_ns() - start) / 1000 / 16)
                assert np.array_equal([r[0] for r in actual], pred[:16])
                assert np.array_equal([r[1] for r in actual], hit[:16])
            latency[mode] = float(np.median(blocks))
            byte_counts[mode] = actual[-1][2]
    return [
        {
            "seed": seed,
            "mode": m,
            "length": length,
            "test_examples": N_TEST,
            "accuracy": float(totals[m][0] / N_TEST),
            "key_hit_or_max_attention": float(totals[m][1] / N_TEST),
            "cache_bytes": byte_counts[m],
            "latency_us": latency[m],
        }
        for m in POLICIES
    ]


if __name__ == "__main__":
    verify_registration()
    recipe = ROOT / "mechanisms" / "recall_extension_protocol.md"
    digest = hashlib.sha256(recipe.read_bytes()).hexdigest()
    assert digest == (OUT / "extension.sha256").read_text().split()[0]
    rows = []
    for seed in SEEDS:
        p, _ = fit(seed, 4)
        for length in (32, 96, 1024):
            rows.extend({**r, "extension_sha256": digest} for r in run(seed, length, p))
        save("memory_extension", rows)
        print("supplementary recall completed", seed, flush=True)
