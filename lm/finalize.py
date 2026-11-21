"""Freeze search, train predefined paired comparisons, and evaluate count baselines."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from prepare import RESULTS, PREFIX, WINDOW, split_data, evaluation_windows
from train import flops

def freeze():
    rows=[json.loads(p.read_text()) for p in sorted((RESULTS/"search").glob("trial-*.decision.json"))]
    if not (len(rows)==8 or (len(rows)>=4 and all(r["status"]=="discard" for r in rows[-3:]))):
        raise RuntimeError("Search stopping rule not satisfied")
    chosen=[r for r in rows if r["status"]=="keep"][-1]
    path=RESULTS/"frozen.json"
    if path.exists():
        if json.loads(path.read_text())!=chosen: raise RuntimeError("Existing selection differs")
    else: path.write_text(json.dumps(chosen,indent=2)+"\n")

def counts():
    train=split_data("train").tobytes()
    counts=[defaultdict(Counter) for _ in range(4)]
    for i,char in enumerate(train):
        for n in range(min(3,i)+1):
            counts[n][train[i-n:i]][char]+=1
    totals=[{k:sum(v.values()) for k,v in group.items()} for group in counts]
    scores={1:0.,3:0.}
    examples=evaluation_windows("test")
    for example in examples:
        s=example.astype("uint8").tobytes()
        for i in range(PREFIX+1,WINDOW+1):
            c=s[i]; p=(counts[0][b""][c]+.5)/(len(train)+128)
            for n in (1,2,3):
                key=s[i-n:i]
                row=counts[n].get(key)
                if row:
                    t=len(row)
                    p=(row[c]+t*p)/(totals[n][key]+t)
                if n in scores: scores[n]-=np.log2(p)
    size=len(examples)*(WINDOW-PREFIX)
    (RESULTS/"counts.json").write_text(json.dumps({"order_is_number_of_prior_bytes":True,"training_bytes":len(train),"test_bytes":size,"bpb":{k:v/size for k,v in scores.items()}},indent=2)+"\n")

def small():
    cfg=json.loads((RESULTS/"frozen.json").read_text())["config"]
    matched=round(320*flops(4,True,True)/flops(2,False,True))
    for seed in (11,23,37):
        variants=[(m,m,320,cfg["lr"]) for m in ["1","2","4","random","adaptive"]]
        variants.append(("2-matched","2",matched,cfg["lr"]))
        if cfg["lr"]!=.002: variants.append(("4-base-lr","4",320,.002))
        for name,mode,steps,lr in variants:
            out=RESULTS/f"final/{name}-{seed}.json"
            if out.exists(): continue
            args=["--steps",str(steps),"--seed",str(seed),"--mode",mode,"--lr",str(lr),"--cost",str(cfg["cost"]),"--bias",str(cfg["bias"]),"--output",str(out)]
            if mode=="adaptive" and seed==11: args+=["--weights",str(RESULTS/"adaptive-11.safetensors")]
            subprocess.run([sys.executable,str(Path(__file__).with_name("train.py")),*args],check=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["freeze","small","counts","prototype"])
    stage=p.parse_args().stage
    if stage=="freeze": freeze()
    elif not (RESULTS/"frozen.json").exists(): raise RuntimeError("Freeze search first")
    elif stage=="small": small()
    elif stage=="counts": counts()
    else:
        for seed in (11,23,37):
            for mode in ("adaptive","4"):
                final=2048 if seed==11 else 512
                if (RESULTS/f"prototype/{mode}-{seed}-{final}.json").exists(): continue
                subprocess.run([sys.executable,str(Path(__file__).with_name("prototype.py")),"--mode",mode,"--seed",str(seed)],check=True)
