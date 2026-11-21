"""Example: .venv/bin/python reasoning/predict.py 1 not zero not --length 64"""
import argparse
import json

import mlx.core as mx
import numpy as np

from common import Decoder, OPS, OUT, QUERY, VOCAB


def predict(bit, operations, length=64, seed=0, checkpoint='adaptive'):
    if bit not in (0,1):
        raise ValueError("Starting bit must be 0 or 1")
    if length < len(operations) + 2:
        raise ValueError("The sequence must have room for all instructions and both endpoints")
    tags = {"pilot":"", "optimized":"optimized-", "adaptive":"qualified-adaptive-", "fixed2":"qualified-fixed2-"}
    tag = tags[checkpoint]
    config = json.loads((OUT / f"{tag}checkpoint.json").read_text())
    model = Decoder(config["width"], config["mode"])
    model.load_weights(str(OUT / f"{tag}reasoning-seed17.safetensors"))
    rng = np.random.default_rng(seed)
    x = rng.integers(7, VOCAB, size=(1,length), dtype=np.int32)
    x[0,0], x[0,-1] = bit, QUERY
    slots = np.sort(rng.choice(length - 2, size=len(operations), replace=False) + 1)
    x[0,slots] = [OPS[op] for op in operations]
    if config["mode"] == "adaptive":
        predicted, depths = model.compact(x, config["threshold"])
    else:
        depth = int(config["mode"][-1])
        logits, _ = model(mx.array(x), depth)
        predicted, depths = np.asarray(logits[:, -1].argmax(-1)), [depth]
    expected = bit
    for op in operations:
        if op == "not":
            expected ^= 1
        elif op in ("zero","one"):
            expected = int(op == "one")
    return {"checkpoint":checkpoint,"starting_bit":bit,"program":operations,"length":length,"model_prediction":int(predicted[0]),
            "correct_answer":expected,"recurrent_passes":int(depths[0])}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bit", type=int, choices=[0,1])
    parser.add_argument("operations", nargs="+", choices=list(OPS))
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", choices=["pilot","optimized","adaptive","fixed2"], default="adaptive", help="Select the 120-update pilot, 2400-update adaptive follow-up, or either 8000-update engineering checkpoint")
    args=parser.parse_args()
    print(json.dumps(predict(args.bit,args.operations,args.length,args.seed,args.checkpoint), indent=2))
