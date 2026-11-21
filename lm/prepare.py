"""Pinned data, disjoint splits and immutable scoring windows for the LM study."""
from pathlib import Path
import hashlib
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/lm"
DATA = ROOT / "lm/data/input.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
DIGEST = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
WINDOW, PREFIX, BATCH, WIDTH, HEADS, VOCAB = 64, 16, 16, 96, 4, 256

def split_data(which):
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Unexpected corpus checksum")
    a, b = len(raw) * 8 // 10, len(raw) * 9 // 10
    lo, hi = {"train": (0, a), "val": (a, b), "test": (b, len(raw))}[which]
    return np.frombuffer(raw[lo:hi], dtype=np.uint8)

def windows_at(data, indices):
    return data[indices[:, None] + np.arange(WINDOW + 1)].astype(np.int32)

def evaluation_windows(which):
    seq = split_data(which)
    starts = np.arange(0, len(seq) - WINDOW, WINDOW - PREFIX)
    if which == "val":
        starts = starts[np.linspace(0, len(starts) - 1, 192, dtype=int)]
    return windows_at(seq, starts)
