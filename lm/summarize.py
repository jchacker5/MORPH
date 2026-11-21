"""Reporting only. Derive all seed uncertainty and cost summaries from raw receipts."""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import t
from prepare import RESULTS

def interval(values):
    x=np.asarray(values,dtype=float)
    mean=float(x.mean())
    radius=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None
    return dict(mean=mean,low=mean-radius if radius is not None else None,high=mean+radius if radius is not None else None,values=x.tolist())

def paired(a,b):
    return interval(np.asarray(a)-np.asarray(b))

def summarize():
    seeds=[11,23,37]
    rows=[]
    designs={}
    def add(name,seed,p):
        j=json.loads(p.read_text())
        r=dict(design=name,seed=seed,parameters=j["parameters"],steps=j["steps"],input_bytes=j["input_bytes"],bpbyte=j["test"]["bpb"],
               byte_perplexity=2**j["test"]["bpb"],mean_depth=j["test"]["mean_depth"],test_windows=j["test"]["examples"],
               scored_test_bytes=j["test"]["scored_bytes"],inference_flops_per_window=j["test"]["inference_flops_per_window"],
               train_flops=j["train_flops"],peak_mlx_train_bytes=j["training_peak_mlx_bytes"],batch1_median_ms=j["batch1_median_ms"])
        rows.append(r)
        designs.setdefault(name,{})[seed]=r
    for p in sorted((RESULTS/"final").glob("*.json")):
        label,seed=p.stem.rsplit("-",1)
        add("small-"+label,int(seed),p)
    for p in sorted((RESULTS/"prototype").glob("*.json")):
        mode,seed,steps=p.stem.split("-")
        add(f"wide-{mode}-{steps}",int(seed),p)
    summary={name:{k:interval([r[k] for r in by_seed.values()]) for k in ["bpbyte","mean_depth","train_flops","inference_flops_per_window","batch1_median_ms","peak_mlx_train_bytes"]} for name,by_seed in designs.items()}
    metrics={}
    for a,b in [("small-adaptive",f"small-{x}") for x in ["1","2","4","random","2-matched"]]+[("wide-adaptive-512","wide-4-512"),("wide-adaptive-2048","wide-4-2048")]:
        if a in designs and b in designs:
            shared=sorted(set(designs[a]) & set(designs[b]))
            metrics[a+" minus "+b]=paired([designs[a][s]["bpbyte"] for s in shared],[designs[b][s]["bpbyte"] for s in shared])
    with (RESULTS/"fits.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (RESULTS/"summary.json").write_text(json.dumps({"by_design":summary,"paired_bpb":metrics},indent=2)+"\n")

if __name__=="__main__": summarize()
