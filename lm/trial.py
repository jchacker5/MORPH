"""Append-only, validation-only autoresearch attempt ledger."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from prepare import RESULTS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--change",choices=["baseline","cost","lr","bias"],required=True)
    p.add_argument("--value",type=float)
    p.add_argument("--hypothesis",required=True)
    a=p.parse_args()
    folder=RESULTS/"search"
    folder.mkdir(exist_ok=True,parents=True)
    previous=sorted(folder.glob("trial-*.decision.json"))
    configs={"lr":.002,"cost":.02,"bias":0.}
    best=float("inf")
    for path in previous:
        row=json.loads(path.read_text())
        if row["status"]=="keep":
            configs,best=row["config"].copy(),row["objective"]
    if (a.change=="baseline") != (len(previous)==0):
        raise ValueError("Baseline must be first and unique")
    if a.change!="baseline":
        configs[a.change]=a.value
    stem=f"trial-{len(previous)+1:02d}"
    result=folder/(stem+".json")
    decision=folder/(stem+".decision.json")
    prerecord=folder/(stem+".plan.json")
    record=dict(config=configs,hypothesis=a.hypothesis,trial=len(previous)+1,source={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in ["train.py","prepare.py"]})
    with prerecord.open("x") as f: json.dump(record,f,indent=2)
    subprocess.run([sys.executable,str(Path(__file__).with_name("train.py")),"--seconds","30","--seed","7","--output",str(result),*[v for k,x in configs.items() for v in ("--"+k,str(x))]],check=True)
    val=json.loads(result.read_text())["val"]
    obj=val["bpb"]+.08*max(0,val["mean_depth"]-2)
    record.update(objective=obj,status="keep" if obj<best-.005 else "discard",val_bpb=val["bpb"],mean_depth=val["mean_depth"])
    decision.write_text(json.dumps(record,indent=2)+"\n")
    ledger=folder/"results.tsv"
    with ledger.open("a") as f:
        writer=csv.writer(f,delimiter="\t")
        if not previous: writer.writerow(["trial","val_bpb","mean_depth","objective","status","hypothesis"])
        writer.writerow([record[k] for k in ["trial","val_bpb","mean_depth","objective","status","hypothesis"]])
    print(record)
if __name__=="__main__": main()
