"""Apply the trained GM classifier to a four-value header, eight symbols and a binary query."""
import argparse
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "results/joint/gm_seed41.npz"
PAIR = np.array([[[0,1],[2,3]],[[0,2],[1,3]]])
EYE = np.eye(4,dtype=np.float32)


def predict(p, header, tokens, query):
    h = np.asarray(header,dtype=np.float32)
    tok = np.asarray(tokens)
    if h.shape!=(4,) or not np.isfinite(h).all():
        raise ValueError("Supply four finite real header features.")
    if tok.shape!=(8,) or not np.issubdtype(tok.dtype,np.integer) or np.any((tok<0)|(tok>3)):
        raise ValueError("Supply eight integer symbols in the range 0 to 3.")
    if not isinstance(query,(int,np.integer)) or query not in (0,1):
        raise ValueError("Supply a query bit of 0 or 1.")
    ctrl = np.tanh(h @ p["c1"]+p["c1b"])
    # Only G and M output columns are needed by this compact policy.
    g,m = (ctrl @ p["c2"][:,[1,3]] + p["c2b"][[1,3]] > 0).astype(int)
    bank = tok[m*4:m*4+4]
    group = EYE[bank[PAIR[g]]].sum(1)[query]
    base = np.tanh(np.concatenate([group,h]) @ p["stem"] + p["sb"])
    z = base
    for _ in range(2):
        z = np.tanh(z @ p["expert"][query]+p["eb"][query,0]+base)
    logits = z @ p["out"]+p["ob"]
    return {"class":int(logits.argmax()),"retained_bank":int(m),"pair_layout":int(g),
            "expert":int(query),"recurrent_steps":2}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header",type=float,nargs=4,required=True)
    parser.add_argument("--tokens",type=int,nargs=8,required=True)
    parser.add_argument("--query",type=int,required=True)
    args=parser.parse_args()
    with np.load(WEIGHTS) as archive:
        weights=dict(archive)
    print(json.dumps(predict(weights,args.header,args.tokens,args.query),indent=2))


if __name__=="__main__":
    main()
