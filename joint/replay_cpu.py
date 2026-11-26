"""Replay the published GM inference receipt with NumPy; no training or MLX."""
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .infer import WEIGHTS, predict
else:
    from infer import WEIGHTS, predict

ROOT = Path(__file__).resolve().parents[1]


def examples(seed, count):
    """The frozen generator in joint/run.py and docs/joint_protocol.md."""
    rng = np.random.default_rng(seed)
    transform = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                          [1, 1, -1, -1], [1, -1, -1, 1]], np.float32) / 2
    layouts = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]]])
    permutations = np.array([[1, 2, 3, 0], [0, 2, 3, 1]])
    bits = rng.integers(0, 2, size=(count, 4))
    header = ((2 * bits - 1) @ transform + rng.normal(0, .15, (count, 4))).astype(np.float32)
    tokens = rng.integers(0, 4, (count, 8))
    query = rng.integers(0, 2, count)
    depth, grouping, expert, memory = bits.T
    addresses = layouts[grouping, query] + 4 * memory[:, None]
    start = np.take_along_axis(tokens, addresses, axis=1).sum(1) % 4
    one = permutations[expert, start]
    labels = np.where(depth, permutations[expert, permutations[expert, one]], one)
    return header, tokens, query, labels


def dense_reference(weights, header, tokens, query):
    """Evaluate every bank/layout/expert, then select the GM policy's path.

    This deliberately does not call predict or reuse its selected-path indexing.
    It is the fixed-depth GM specialization of the frozen dense study reference.
    """
    layouts = (((0, 1), (2, 3)), ((0, 2), (1, 3)))
    choices = []
    for bank in (0, 1):
        for layout in layouts:
            indices = np.array(layout)[query] + 4 * bank
            selected = np.take_along_axis(tokens, indices, axis=1)
            grouped = np.eye(4, dtype=np.float32)[selected].sum(1)
            base = np.tanh(np.concatenate([grouped, header], axis=1) @ weights["stem"] + weights["sb"])
            experts = []
            for expert in (0, 1):
                state = base
                for _ in range(2):
                    state = np.tanh(state @ weights["expert"][expert] + weights["eb"][expert] + base)
                experts.append(state @ weights["out"] + weights["ob"])
            choices.append(np.stack(experts, axis=1))
    scores = np.stack(choices, axis=1)
    control = np.tanh(header @ weights["c1"] + weights["c1b"]) @ weights["c2"] + weights["c2b"]
    path = 2 * (control[:, 3] > 0) + (control[:, 1] > 0).astype(int)
    return scores[np.arange(len(query)), path, query].argmax(-1)


def replay():
    receipt = json.loads((ROOT / "results/joint/usability.json").read_text())
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if digest != receipt["checkpoint_sha256"]:
        raise ValueError("The frozen checkpoint checksum does not match its receipt")
    protocol = hashlib.sha256((ROOT / "joint/usage_check.md").read_bytes()).hexdigest()
    if protocol != receipt["usage_protocol_sha256"]:
        raise ValueError("The frozen usage protocol checksum does not match its receipt")
    with np.load(WEIGHTS, allow_pickle=False) as archive:
        weights = dict(archive)
    header, tokens, query, labels = examples(receipt["fresh_seed"], receipt["fresh_n"])
    predicted = np.array([predict(weights, h, t, q)["class"] for h, t, q in zip(header, tokens, query)])
    count = int((predicted == labels).sum())
    if count != receipt["fresh_correct"]:
        raise ValueError(f"Frozen replay changed: {count}/{len(labels)}, expected {receipt['fresh_correct']}")
    n = receipt["single_vs_batched_matches"]
    np.testing.assert_array_equal(predicted[:n], dense_reference(weights, header[:n], tokens[:n], query[:n]))
    return {"checkpoint_sha256": digest, "examples": len(labels), "correct": count,
            "accuracy": count / len(labels), "dense_reference_matches": n,
            "numpy_version": np.__version__}


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
