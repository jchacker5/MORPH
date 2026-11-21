"""Summarize paired independent shallow controls and the primary adaptive/four-pass fits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/prototype_controls"
PRIMARY=ROOT/"results/lm/prototype"
SEEDS=(11,23,37)


def get(path):
    return json.loads(path.read_text())


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def row(seed,step,mode,subset,score,train):
    return {"seed":seed,"steps":step,"model":mode,"subset":subset,
            "bpb":score["bpb"],"byte_perplexity":2**score["bpb"],
            "scored_bytes":score["scored_bytes"],"hard_depth":score["mean_depth"],
            "exit_histogram":"|".join(str(v) for v in score["histogram"]),
            "estimated_inference_flops_per_window":score["inference_flops_per_window"],
            "estimated_train_flops":train["train_flops"],"input_bytes":train["input_bytes"],
            "parameters":train["parameters"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available",action="store_true",help="Write clearly named interim tables while primary fits finish.")
    interim=parser.parse_args().available
    prefix="interim_" if interim else ""
    rows=[]
    for steps,seeds in ((512,SEEDS),(2048,(11,))):
        for seed in seeds:
            for d in (1,2):
                path=OUT/f"fixed{d}-{seed}-{steps}.json"
                if interim and not path.exists(): continue
                train=get(path)
                for s,score in train["scores"].items():
                    rows.append(row(seed,steps,f"fixed{d}",s,score,train))
            for name,src in (("adaptive","adaptive"),("fixed4","4")):
                path=PRIMARY/f"{src}-{seed}-{steps}.json"
                if interim and not path.exists(): continue
                train=get(path)
                rows.append(row(seed,steps,name,"plays",train["test"],train))
            if seed==11:
                path=OUT/f"reloaded-adaptive-11-{steps}.json"
                if interim and not path.exists(): continue
                replay=get(path)
                rows.append(row(seed,steps,"adaptive","sonnets",replay["scores"]["sonnets"],replay))
    for record in rows:
        counts=[int(x) for x in record["exit_histogram"].split("|")]
        assert sum(counts)*48==record["scored_bytes"]
        assert record["scored_bytes"]=={"plays":111504,"sonnets":87744}[record["subset"]]
        assert np.isclose(np.dot(counts,np.arange(1,5))/sum(counts), record["hard_depth"])
        assert record["input_bytes"]==record["steps"]*4*64
        assert record["parameters"]==(30321217 if record["model"]=="adaptive" else 30319488)
    write_csv(OUT/(prefix+"all_scores.csv"),rows)
    lookup={(r["seed"],r["steps"],r["model"],r["subset"]):r for r in rows}
    summary=[]
    for steps in (512,2048):
        for subset in ("plays","sonnets"):
            for d in (1,2,4):
                paired=[]
                model=f"fixed{d}"
                for seed in SEEDS:
                    a=lookup.get((seed,steps,"adaptive",subset))
                    f=lookup.get((seed,steps,model,subset))
                    if a is not None and f is not None: paired.append((a,f))
                if not paired: continue
                av,fv=zip(*paired)
                diff=np.asarray([a["bpb"]-f["bpb"] for a,f in paired])
                radius=t.ppf(.975,2)*diff.std(ddof=1)/np.sqrt(3) if len(paired)==3 else None
                summary.append({"steps":steps,"subset":subset,"comparison":"adaptive_minus_"+model,
                    "paired_seeds":"|".join(str(r["seed"]) for r in av),"n":len(paired),
                    "adaptive_bpb":np.mean([r["bpb"] for r in av]),"fixed_bpb":np.mean([r["bpb"] for r in fv]),
                    "mean_delta_bpb":diff.mean(),"lo95":diff.mean()-radius if radius is not None else None,
                    "hi95":diff.mean()+radius if radius is not None else None,
                    "adaptive_hard_depth":np.mean([r["hard_depth"] for r in av]),"fixed_hard_depth":d,
                    "mean_adaptive_over_fixed_inference_flops":np.mean([a["estimated_inference_flops_per_window"]/f["estimated_inference_flops_per_window"] for a,f in paired]),
                    "adaptive_over_fixed_training_flops":av[0]["estimated_train_flops"]/fv[0]["estimated_train_flops"]})
    write_csv(OUT/(prefix+"paired_bpb.csv"),summary)
    if not interim:
        assert len(rows)==26 and len(summary)==10
        report(summary)



def report(summary):
    lines=["# Independent shallow-decoder results", "", "Joseph Defendre", "",
           "Six separately trained fixed-depth 30M models produced eight endpoints. All seeds shared the corresponding adaptive model’s initial backbone, training windows and optimizer schedule. The play test covered 111,504 scored bytes and the Sonnets test covered 87,744. All comparisons below are paired on the displayed seeds. Positive BPB differences favor the fixed decoder.", "",
           "| Updates | Corpus | Fixed passes | Paired seeds | Adaptive BPB | Fixed BPB | Difference in BPB [95% interval] | Adaptive / fixed inference FLOPs | Adaptive / fixed training FLOPs |",
           "|---:|---|---:|---|---:|---:|---|---:|---:|"]
    for r in summary:
        ci=(f" [{r['lo95']:+.4f}, {r['hi95']:+.4f}]" if r["n"]==3 else "")
        seeds=r["paired_seeds"].replace("|",", ")
        d=r["fixed_hard_depth"]
        lines.append(f"| {r['steps']} | {r['subset']} | {d} | {seeds} | {r['adaptive_bpb']:.4f} | {r['fixed_bpb']:.4f} | {r['mean_delta_bpb']:+.4f}{ci} | {r['mean_adaptive_over_fixed_inference_flops']:.3f}× | {r['adaptive_over_fixed_training_flops']:.3f}× |")
    lines.extend(["", "The 512-update play comparisons have paired two-sided Student intervals over three training seeds. The external comparisons use the single available adaptive seed. All protocols chose checkpoints by step count before evaluating these controls. Model-level scores and exact histograms are in `all_scores.csv`. The SHA-256 and original-test comparisons for the two adaptive checkpoint reloads are recorded separately.", ""])
    (OUT/"findings.md").write_text("\n".join(lines))


if __name__=="__main__":
    main()
