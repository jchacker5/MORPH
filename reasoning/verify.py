"""Exercise generator semantics, early-exit execution, int64 arithmetic, frozen splits and checkpoints."""
import argparse
import csv
import gzip
import hashlib
import json

import mlx.core as mx
import numpy as np

from check_costs import audit
from common import Decoder, OUT, forward_flops, generate
from run import condition


def interpret(row):
    value = row[0]
    for symbol in row[1:-1]:
        if symbol == 3:
            value ^= 1
        elif symbol in (4,5):
            value = symbol - 4
    return int(value)


def unit():
    x, y, c = generate(np.random.default_rng(31), 48, 64, (1,6))
    assert [interpret(row) for row in x] == list(y)
    assert np.array_equal(((x[:,1:-1] >= 2)&(x[:,1:-1] <= 5)).sum(1), c)
    assert all(row[-1]==6 for row in x)
    flops = forward_flops(2880,32,np.array([1,4],dtype=np.int32),True)
    expected = np.array([d * (24*32*2880**2+4*32**2*2880)+4*2880+2*2880*min(d,3) for d in [1,4]], dtype=np.int64)
    assert flops.dtype == np.int64 and np.array_equal(flops,expected)
    mx.random.seed(31)
    m = Decoder(64,"adaptive")
    logits,_=m(mx.array(x),4,all_exits=True)
    forced=np.asarray(logits).argmax(-1)
    allocate=np.tile(np.arange(1,5),12)
    compact, dep=m.compact(x,allocated=allocate)
    assert np.array_equal(dep,allocate)
    assert np.array_equal(compact, forced[np.arange(len(x)),allocate-1])
    # Check that an update to future positions cannot influence an earlier feature.
    x2=x.copy()
    x2[:,48:]=7
    layer=m.blocks[0]
    prefix1=np.asarray(layer(m.embedding(mx.array(x))))[:,:40]
    prefix2=np.asarray(layer(m.embedding(mx.array(x2))))[:,:40]
    assert np.allclose(prefix1,prefix2,atol=1e-5)


def verify_logs():
    for receipt in ("protocol-freeze", "optimization-freeze", "extension-freeze", "usability-freeze", "sensitivity-freeze"):
        digest, relative = (OUT / f"{receipt}.txt").read_text().splitlines()[1].split(maxsplit=1)
        path=OUT.parents[1] / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, path
    audit()
    counts={"main":0,"scale":0,"long":0,"opt":0,"usable":0}
    for path in sorted(OUT.glob("*-w*.json")):
        stage=path.name.split("-")[0]
        if stage not in counts:
            continue
        rec=json.loads(path.read_text())
        assert np.isfinite(rec["train_objective"]).all()
        with gzip.open(path.with_suffix(".csv.gz"),"rt") as f:
            r=list(csv.DictReader(f))
        for idx,s in enumerate(rec["splits"]):
            draws=[a for a in r if a["split"]==s["name"]]
            shift=(4,6) if s["name"]=="composition64" else (1,3)
            offset={"long":908173,"opt":291105,"usable":817239}.get(stage,0)
            _, labels, count=condition(663000+idx+offset,s["length"],s["n"],shift)
            assert np.array_equal(labels,[int(a["label"]) for a in draws]), path
            assert np.array_equal(count,[int(a["instructions"]) for a in draws]), path
            assert s["correct"]==sum(a["prediction"]==a["label"] for a in draws)
            assert s["depth_hist"]==np.bincount([int(a["passes"]) for a in draws],minlength=5)[1:].tolist()
        counts[stage]+=1
    print("Verified run counts:",counts)
    assert counts=={"main":21,"scale":18,"long":45,"opt":21,"usable":2}
    for tag,archive,n,offset in [("","main-w64-adaptive-s17",192,0),("optimized-","opt-w64-adaptive-s17",192,291105),
                                 ("qualified-adaptive-","usable-w64-adaptive-s17",384,817239),
                                 ("qualified-fixed2-","usable-w64-fixed2-s17",384,817239)]:
        config=json.loads((OUT / f"{tag}checkpoint.json").read_text())
        model=Decoder(config["width"],config["mode"])
        model.load_weights(str(OUT / f"{tag}reasoning-seed17.safetensors"))
        with gzip.open(OUT / f"{archive}.csv.gz","rt") as file:
            expected=list(csv.DictReader(file))[:16]
        # Recreate the entire split; generator sampling includes batch-shaped operations.
        x,_,_=condition(663000+offset,32,n)
        p,d=model.compact(x[:16],threshold=config["threshold"],allocated=None if config["mode"]=="adaptive" else np.full(16,2))
        assert np.array_equal(p,[int(a["prediction"]) for a in expected])
        assert np.array_equal(d,[int(a["passes"]) for a in expected])
    print("All four checkpoints reproduce archived test predictions and hard exits.")
    diagnostics=sorted(OUT.glob("sensitivity-*-w*.json"))
    assert len(diagnostics)==12
    for path in diagnostics:
        result=json.loads(path.read_text())
        with gzip.open(path.with_suffix(".csv.gz"),"rt") as f:
            draws=list(csv.DictReader(f))
        for d in draws:
            cost=int(forward_flops(result["width"],int(d["length"]),int(d["passes"]),True))
            assert cost==int(d["matmul_flops"]), path
        assert sum(s["correct"] for s in result["splits"])==sum(d["prediction"]==d["label"] for d in draws), path
    print("Verified 12 development-threshold sensitivity records.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    unit()
    if args.full:
        verify_logs()
    print("Reasoning verification passed.")
