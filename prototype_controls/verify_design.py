"""Verify paired initialization and fingerprint sampled training indices without training or scoring."""
import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"lm"))
from prototype import B,WideDecoder
from prepare import WINDOW,split_data


def main():
    rows=[]
    n=len(split_data("train"))
    for seed in (11,23,37):
        mx.random.seed(seed)
        static=dict(tree_flatten(WideDecoder(False).parameters()))
        mx.eval(list(static.values()))
        mx.random.seed(seed)
        adaptive=dict(tree_flatten(WideDecoder(True).parameters()))
        mx.eval(list(adaptive.values()))
        assert set(adaptive)-set(static)=={"halt.weight","halt.bias"}
        assert all(bool(mx.array_equal(v,adaptive[k]).item()) for k,v in static.items())
        rng=np.random.default_rng(seed)
        end=2048 if seed==11 else 512
        starts=np.stack([rng.integers(0,n-WINDOW,size=B) for _ in range(end)])
        indices={str(h):hashlib.sha256(starts[:h].astype("<i8").tobytes()).hexdigest() for h in ([512,2048] if seed==11 else [512])}
        rows.append({"seed":seed,"initial_backbone_tensors_identical":len(static),"controller_only_tensors":2,
                     "identical_backbone_parameters":sum(v.size for v in static.values()),
                     "start_indices_sha256_little_endian_int64":indices})
        del static,adaptive
    out=ROOT/"results/prototype_controls/initialization.json"
    out.write_text(json.dumps(rows,indent=2)+"\n")
    print("Paired initialization verified across all seeds.")


if __name__=="__main__":
    main()
