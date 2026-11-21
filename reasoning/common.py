"""Shared generator and MLX decoder for the MORPH Boolean program research task."""
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

OUT = Path(__file__).resolve().parents[1] / "results" / "reasoning"
SEEDS = (17, 29, 43)
ID_LENGTHS = (32, 64, 128)
VOCAB = 16
QUERY = 6
OPS = {"identity": 2, "not": 3, "zero": 4, "one": 5}


def generate(rng, batch, length, depth_range=(1, 3)):
    """Return tokens, labels and independently sampled instruction counts."""
    tokens = rng.integers(7, VOCAB, size=(batch, length), dtype=np.int32)
    values = rng.integers(0, 2, size=batch, dtype=np.int32)
    tokens[:, 0] = values
    tokens[:, -1] = QUERY
    labels = values.copy()
    counts = rng.integers(depth_range[0], depth_range[1] + 1, size=batch)
    for row, count in enumerate(counts):
        slots = np.sort(rng.choice(length - 2, size=count, replace=False) + 1)
        ops = rng.integers(2, 6, size=count)
        tokens[row, slots] = ops
        for op in ops:
            if op == 3:
                labels[row] ^= 1
            elif op >= 4:
                labels[row] = op - 4
    return tokens, labels, counts


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head_dim = min(64, width)
        self.heads = width // self.head_dim
        self.norm1 = nn.RMSNorm(width)
        self.norm2 = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, width * 4, bias=False)
        self.down = nn.Linear(width * 4, width, bias=False)
        self.rope = nn.RoPE(self.head_dim)

    def __call__(self, x):
        batch, length, _ = x.shape
        q, k, v = mx.split(self.qkv(self.norm1(x)), 3, axis=-1)
        def heads(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = self.rope(heads(q)), self.rope(heads(k)), heads(v)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask="causal")
        hidden = x + self.out(attn.transpose(0, 2, 1, 3).reshape(batch, length, -1))
        return hidden + self.down(nn.gelu(self.up(self.norm2(hidden))))


class Decoder(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.width = width
        self.mode = mode
        self.embedding = nn.Embedding(VOCAB, width)
        self.blocks = [Block(width) for _ in range(4 if mode == "untied4" else 1)]
        self.norm = nn.RMSNorm(width)
        self.readout = nn.Linear(width, 2)
        if mode == "adaptive":
            self.hazard = nn.Linear(width, 1)

    def step(self, x, step):
        return self.blocks[step if self.mode == "untied4" else 0](x)

    def __call__(self, tokens, passes=4, all_exits=False):
        x = self.embedding(tokens)
        logits, hazards = [], []
        for i in range(passes):
            x = self.step(x, i)
            if all_exits or i == passes - 1:
                q = self.norm(x[:, -1])
                logits.append(self.readout(q).astype(mx.float32))
                if self.mode == "adaptive" and i < 3:
                    hazards.append(mx.sigmoid(self.hazard(q).squeeze(-1).astype(mx.float32)))
        return mx.stack(logits, 1), mx.stack(hazards, 1) if hazards else None

    def compact(self, tokens, threshold=None, allocated=None):
        """Execute only active samples; stop computing when the last sample exits."""
        x = self.embedding(mx.array(tokens))
        active = np.arange(len(tokens))
        predictions = np.empty(len(tokens), dtype=np.int32)
        depths = np.empty(len(tokens), dtype=np.int32)
        for step in range(1, 5):
            x = self.step(x, step - 1)
            q = self.norm(x[:, -1])
            if allocated is not None:
                halt = allocated[active] == step
            elif step < 4:
                hazard = mx.sigmoid(self.hazard(q).squeeze(-1).astype(mx.float32))
                halt = np.asarray(hazard) >= threshold
            else:
                halt = np.ones(len(active), dtype=bool)
            if np.any(halt):
                idx = mx.array(np.flatnonzero(halt).astype(np.int32))
                predictions[active[halt]] = np.asarray(self.readout(q[idx]).argmax(-1))
                depths[active[halt]] = step
            if halt.all():
                break
            if halt.any():
                x = x[mx.array(np.flatnonzero(~halt).astype(np.int32))]
                active = active[~halt]
        return predictions, depths

    def parameter_count(self):
        return sum(v.size for _, v in tree_flatten(self.trainable_parameters()))


def loss(model, tokens, labels, passes):
    adaptive = model.mode == "adaptive"
    logits, hazards = model(tokens, passes, all_exits=adaptive)
    if not adaptive:
        return nn.losses.cross_entropy(logits[:, -1], labels, reduction="mean")
    survival = mx.ones_like(hazards[:, 0])
    parts = []
    for t in range(3):
        p = survival * hazards[:, t]
        parts.append(p * (nn.losses.cross_entropy(logits[:, t], labels) + 0.012 * (t + 1)))
        survival = survival * (1.0 - hazards[:, t])
    parts.append(survival * (nn.losses.cross_entropy(logits[:, 3], labels) + 0.012 * 4))
    return sum(parts).mean()


def forward_flops(width, length, depth, adaptive=False, train=False):
    # Dense math reference; triangle-aware fused kernels can do less attention arithmetic.
    block = 24 * length * width**2 + 4 * length**2 * width
    depth = np.asarray(depth, dtype=np.int64)
    total = depth * block
    if adaptive:
        total += (depth if train else 1) * 4 * width + np.minimum(depth, 3) * 2 * width
    else:
        total += 4 * width
    return total
