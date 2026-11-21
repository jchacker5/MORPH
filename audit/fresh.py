"""Compare independent reruns using deterministic task endpoints, excluding timing."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def table(folder,file,keys):
    with (folder/"results"/file).open() as f:
        return {tuple(row[k] for k in keys):row for row in csv.DictReader(f)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("fresh",type=Path)
    args=p.parse_args()
    results={}
    specs=[
        ("joint/fits.csv",["seed","cell"],["accuracy","cross_entropy","mean_passes","compact_inference_linear_macs"]),
        ("mechanisms/grouping.csv",["seed","mode","slots"],["accuracy","logloss","ari","max_suffix_delta"]),
        ("mechanisms/routing.csv",["seed","mode"],["accuracy","active_params"]),
        ("mechanisms/memory.csv",["seed","mode","slots","condition","length"],["accuracy","key_hit","cache_bytes"])
    ]
    for file,keys,metrics in specs:
        orig=table(ROOT,file,keys)
        fresh=table(args.fresh,file,keys)
        if orig.keys()!=fresh.keys(): raise ValueError(f"Incomplete replication for {file}")
        deltas={k:max(abs(float(row[k])-float(fresh[id_][k])) for id_,row in orig.items()) for k in metrics}
        if max(deltas.values())>1e-6: raise ValueError(f"{file}: {deltas}")
        results[file]={"rows":len(orig),"max_abs_difference":deltas}
    (ROOT/"results/fresh_replication.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
