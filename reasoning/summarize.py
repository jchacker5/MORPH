"""Compute paired seed summaries and the attained fixed-policy compute frontier."""
import argparse
import csv
import gzip
import itertools
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, t

from common import OUT, SEEDS


def interval(vals):
    vals = np.asarray(vals, dtype=np.float64)
    mean = vals.mean()
    radius = t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/len(vals)**.5 if len(vals) > 1 else 0
    return {"mean":float(mean),"lo":float(mean-radius),"hi":float(mean+radius),"values":list(vals)}


def frontier(points, budget):
    choices = [(p["accuracy"],p["mode"]) for p in points if p["flops"] <= budget]
    exact = [(p["accuracy"],p["mode"]) for p in points if p["flops"] == budget]
    for a,b in itertools.permutations(points, 2):
        if a["flops"] < budget < b["flops"]:
            q = (budget-a["flops"])/(b["flops"]-a["flops"])
            entry = ((1-q)*a["accuracy"]+q*b["accuracy"],f"{a['mode']}({1-q:.4f})+{b['mode']}({q:.4f})")
            choices.append(entry)
            exact.append(entry)
    return (max(choices) if choices else (None,None)), (max(exact) if exact else (None,None))


def design(prefix, size=None):
    metrics = []
    for path in sorted(OUT.glob(f"{prefix}-w*-s*.json")):
        rec = json.loads(path.read_text())
        if size and size != rec["width"]:
            continue
        with gzip.open(path.with_suffix(".csv.gz"),"rt") as f:
            draws = list(csv.DictReader(f))
        for tag in ["id"] + [r["name"] for r in rec["splits"]]:
            keep = lambda name: name.startswith("id") if tag == "id" else name == tag
            splits = [r for r in rec["splits"] if keep(r["name"])]
            samples = [r for r in draws if keep(r["split"])]
            n = sum(r["n"] for r in splits)
            avg = lambda key: sum(r["n"]*r[key] for r in splits)/n
            sm = {"seed":rec["seed"],"mode":rec["mode"],"group":tag,"n":n,"width":rec["width"],
                  "accuracy":sum(r["correct"] for r in splits)/n,"flops":avg("mean_flops_est"),
                  "depth":avg("mean_depth"),"params":rec["parameters"],"train_flops":rec["train_flops_est"],
                  "train_tokens":rec["train_tokens"],"train_seconds":sum(rec["train_step_seconds"]),
                  "ms_per_example":1000 * sum(r["inference_seconds"] for r in splits)/n,
                  "peak_inference_mib":max(r["physical_inference_peak_mlx_bytes"] for r in splits)/2**20,
                  "peak_train_mib":rec["train_peak_mlx_bytes"]/2**20}
            if sm["mode"] == "adaptive":
                reference = sum(np.array(r["permuted_depth_correct"]) for r in splits)/n
                forced = sum(np.array(r["force_depth_correct"]) for r in splits)/n
                depth = [int(r["passes"]) for r in samples]
                difficulty = [int(r["instructions"]) for r in samples]
                pred_dis = sum(r["cache_prediction_disagreement"] for r in splits)
                depth_dis = sum(r["cache_depth_disagreement"] for r in splits)
                sm.update({"shuffle_accuracy":reference.mean(),"shuffle_gain_pp":100*(sm["accuracy"]-reference.mean()),
                           "raw_accuracy":sum(r["raw_correct"] for r in splits)/n,"raw_depth":avg("raw_mean_depth"),
                           "depth_count_rho":float(spearmanr(depth,difficulty).statistic) if len(set(depth))>1 else None,
                           "cache_prediction_disagreement":pred_dis,"cache_depth_disagreement":depth_dis,
                           "depth_hist":sum(np.array(r["depth_hist"]) for r in splits).tolist(),
                           "threshold":rec["threshold"],"dev_depth":rec["dev_depth"],
                           **{f"forced{d}_accuracy":v for d,v in enumerate(forced,1)}})
            metrics.append(sm)
    groups = sorted(set(m["group"] for m in metrics))
    output = {"model":{},"adaptive_comparison":{}}
    for group in groups:
        subset = [m for m in metrics if m["group"] == group]
        for mode in sorted(set(m["mode"] for m in subset)):
            selected = sorted((m for m in subset if m["mode"] == mode),key=lambda m:m["seed"])
            if len(selected) != 3:
                raise ValueError(f"{prefix}/{size}/{mode} incomplete")
            keys = ["accuracy","flops","depth","params","train_flops","train_tokens","train_seconds",
                    "ms_per_example","peak_inference_mib","peak_train_mib"]
            if mode == "adaptive":
                keys += ["shuffle_accuracy","shuffle_gain_pp","raw_accuracy","raw_depth","cache_prediction_disagreement",
                        "cache_depth_disagreement","dev_depth","forced1_accuracy","forced2_accuracy","forced3_accuracy","forced4_accuracy"]
            reporting = {k:interval([m[k] for m in selected]) for k in keys}
            if mode == "adaptive":
                reporting["depth_count_rho"] = [m["depth_count_rho"] for m in selected]
                reporting["depth_hist_by_seed"] = [m["depth_hist"] for m in selected]
                reporting["threshold_by_seed"] = [m["threshold"] for m in selected]
            output["model"].setdefault(mode,{})[group] = reporting
        paired = []
        for seed in SEEDS:
            eligible = [m for m in subset if m["seed"]==seed]
            adaptive = next(m for m in eligible if m["mode"]=="adaptive")
            fixed = [m for m in eligible if m["mode"].startswith(("fixed","untied"))]
            bound, exact = frontier(fixed,adaptive["flops"])
            shared = next(m for m in fixed if m["mode"]=="fixed4")
            paired.append({"seed":seed,"best_fixed_accuracy":bound[0],"best_fixed_policy":bound[1],
                           "frontier_gain_pp":100*(adaptive["accuracy"]-bound[0]) if bound[0] is not None else None,
                           "exact_mix_gain_pp":100*(adaptive["accuracy"]-exact[0]) if exact[0] is not None else None,
                           "exact_mix":exact[1],"fixed4_gain_pp":100*(adaptive["accuracy"]-shared["accuracy"]),
                           "training_cost_relative_gap":adaptive["train_flops"]/shared["train_flops"]-1})
        report={"rows":paired}
        for key in ["frontier_gain_pp","exact_mix_gain_pp","fixed4_gain_pp","training_cost_relative_gap"]:
            values=[r[key] for r in paired if r[key] is not None]
            report[key]=interval(values) if len(values)==3 else None
        output["adaptive_comparison"][group]=report
    return output,metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("suite",choices=["main","scale","opt","long"])
    args=parser.parse_args()
    allrows=[]
    result={}
    for width in ((896,1600,2880) if args.suite in ("scale","long") else (None,)):
        res, rows=design(args.suite,width)
        result[str(width or "pilot")]=res
        allrows.extend(rows)
    (OUT / f"{args.suite}-summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    fields = sorted(set().union(*(r.keys() for r in allrows)))
    with (OUT / f"{args.suite}-metrics.csv").open("w") as out:
        writer=csv.DictWriter(out,fieldnames=fields)
        writer.writeheader()
        writer.writerows(allrows)
    for size,res in result.items():
        print(size, "Adaptive ID",res["model"]["adaptive"]["id"]["accuracy"], "fixed frontier difference",
              res["adaptive_comparison"]["id"]["frontier_gain_pp"])


if __name__=="__main__":
    main()
