"""Exact breakpoints on the development set, preserving original train runs and held-out tests."""
import csv
import gzip
import itertools
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from common import Decoder, ID_LENGTHS, OUT, SEEDS
from run import cached, condition, hard_depth, score, train


def exact_threshold(hazards):
    # The >= comparison changes immediately above each observed hazard.
    boundaries=np.unique(hazards).astype(np.float64)
    after=np.nextafter(boundaries, np.inf)
    candidates=np.unique(np.concatenate([[0.0,1.0],after[after <= 1.0]]))
    # Compare in float64 so the selected breakpoint is not rounded back to the float32 observation.
    h=hazards.astype(np.float64)
    best=min(candidates,key=lambda t: abs(hard_depth(h,t).mean()-2.5))
    # Physical inference compares float32 hazards; use a representable threshold immediately above the boundary.
    physical=np.float32(best)
    if float(physical) < best:
        physical=np.nextafter(physical,np.float32(np.inf))
    assert np.array_equal(hard_depth(h,best),hard_depth(hazards,float(physical)))
    return float(physical),float(hard_depth(hazards,float(physical)).mean()),len(candidates)


def write(model, seed, extended, optimized, orig_rows, original_threshold=None):
    stage="long" if extended else "opt"
    dev_h=[]
    for length in ((32,) if extended else ID_LENGTHS):
        x,_,_=condition(82406+length+(379900 if extended else 205089),length,128 if extended else 96)
        _,h=cached(model,x,4 if extended else 16)
        dev_h.append(h)
    threshold,dev_depth,breakpoints=exact_threshold(np.concatenate(dev_h))
    splits=[]
    record=[]
    for code,length in enumerate((32,) if extended else ID_LENGTHS):
        n=256 if extended else 192
        x,y,c=condition(663000+code+(908173 if extended else 291105),length,n)
        splits.append(score(model,f"id{length}",x,y,c,threshold,extended,record,code))
    # Raw per-step logits may change by machine/precision/replay. Preserve explicit audit.
    original=[r for r in orig_rows if str(r[0]).startswith("id")]
    if len(original)!=len(record):
        raise ValueError("Unexpected ID example count")
    ref_forced=np.array([r[8:12] for r in original],dtype=int)
    ref_h=np.array([r[12:15] for r in original],dtype=float)
    new_forced=np.array([r[8:12] for r in record],dtype=int)
    new_h=np.array([r[12:15] for r in record],dtype=float)
    if original_threshold is None:
        orig_report=json.loads((OUT / f"{stage}-w{model.width}-adaptive-s{seed}.json").read_text())
        original_threshold=orig_report["threshold"]
    replay_d=hard_depth(new_h.astype(np.float32),original_threshold)
    original_d=np.array([r[6] for r in original],dtype=int)
    original_pred=np.array([r[5] for r in original],dtype=int)
    stats={"seed":seed,"width":model.width,"stage":stage,"refined_threshold":threshold,"dev_depth":dev_depth,
           "breakpoint_count":breakpoints,"n_id":len(record),
           "replayed_forced_predictions_changed":int((ref_forced!=new_forced).sum()),
           "replayed_max_abs_hazard_change":float(np.max(np.abs(ref_h-new_h))),
           "original_policy_vs_replay_cached_depth_changes":int((original_d!=replay_d).sum()),
           "original_policy_vs_replay_cached_prediction_changes":int((original_pred!=new_forced[np.arange(len(replay_d)),replay_d-1]).sum()),
           "splits":splits}
    name=f"sensitivity-{stage}-w{model.width}-adaptive-s{seed}"
    with gzip.open(OUT / (name+".csv.gz"),"wt") as f:
        w=csv.writer(f)
        w.writerow(["split","example","length","instructions","label","prediction","passes","matmul_flops",
                    *[f"forced_prediction_{i}" for i in range(1,5)],*[f"halt_probability_{i}" for i in range(1,4)]])
        w.writerows(record)
    tmp=OUT / (name+".tmp")
    tmp.write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n")
    tmp.replace(OUT / (name+".json"))


def replay():
    for stage,width,seed in itertools.chain(itertools.product(["long"],[896,1600,2880],SEEDS),
                                             itertools.product(["opt"],[64],SEEDS)):
        dest=OUT/f"sensitivity-{stage}-w{width}-adaptive-s{seed}.json"
        if dest.exists():
            continue
        base=OUT/f"{stage}-w{width}-adaptive-s{seed}.json"
        if not base.exists():
            continue
        with gzip.open(base.with_suffix(".csv.gz"),"rt") as f:
            r=csv.reader(f);next(r);original=list(r)
        extended=stage=="long"
        mx.random.seed(seed)
        model=Decoder(width,"adaptive")
        if extended:
            model.set_dtype(mx.bfloat16)
        if stage=="opt" and seed==17:
            model.load_weights(str(OUT/"optimized-reasoning-seed17.safetensors"))
        else:
            train(model,seed,160 if extended else 2400,extended,extended,not extended)
        write(model,seed,extended,not extended,original)
        print(dest,flush=True)
        del model
        mx.clear_cache()


if __name__=="__main__":
    replay()
