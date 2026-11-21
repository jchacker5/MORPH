"""Held-out Project Gutenberg sonnets at pinned GITenberg revision."""
import hashlib
from pathlib import Path
import re
import urllib.request

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
URL="https://raw.githubusercontent.com/GITenberg/Shakespeare-s-Sonnets_1041/b6da8a3e73e7081cf71c57e930001d0b92a8f644/1041.txt"
DIGEST="e3580b25135e2d89467dd4cef0ccdcdd5f50f8779247fc24612ef653d27dc9d5"
def poems():
    source=ROOT/"lm_followup/data/source.txt"
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_bytes(urllib.request.urlopen(URL,timeout=30).read())
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DIGEST: raise ValueError("External corpus hash mismatch")
    lines=raw.decode("ascii").splitlines()
    first=lines.index("  I")
    last=lines.index("    Love's fire heats water, water cools not love.")
    result=[]
    for ln in lines[first:last+1]:
        content=ln.strip()
        if re.fullmatch("[IVXLCDM]+",content): result.append([])
        elif content: result[-1].append(content)
    if len(result)!=154: raise ValueError("Unexpected sonnet count")
    return [("\n".join(p)+"\n").encode("ascii") for p in result]

def eval_windows():
    from prepare import WINDOW,PREFIX,windows_at,split_data
    windows=np.concatenate([windows_at(np.frombuffer(p,dtype=np.uint8),np.arange(0,len(p)-WINDOW,WINDOW-PREFIX)) for p in poems()])
    # Exclude exact leakage; check every full evaluation context against training.
    train=split_data("train").tobytes()
    duplicates=sum(bytes(row.astype("uint8")) in train for row in windows)
    if duplicates: raise ValueError(f"Found {duplicates} external context matches in LM training bytes")
    return windows
