"""Compare the refined-threshold model to the original fixed policies at its realized budget."""
import csv
import gzip
import json
import numpy as np
from common import OUT, SEEDS, forward_flops
from run import hard_depth
from sensitivity import exact_threshold
from summarize import frontier,interval


def archived_half(stage,width,seed):
    # A post hoc alternative limited to retained outputs from the exact original weights.
    points=[]
    learned={}
    for file in OUT.glob(f"{stage}-w*-s{seed}.csv.gz"):
        _,w,mode,_=file.name.split("-")
        w=int(w[1:])
        if stage=="long" and w!=width:
            continue
        if mode=="random":
            continue
        with gzip.open(file,"rt") as handle:
            draws=[r for r in csv.DictReader(handle) if r["split"].startswith("id")]
        lengths=np.array([int(r["length"]) for r in draws])
        selected=np.zeros(len(draws),dtype=bool)
        for length in np.unique(lengths):
            indices=np.flatnonzero(lengths==length)
            selected[indices[len(indices)//2:]]=True
        labels=np.array([int(r["label"]) for r in draws])
        if mode=="adaptive":
            hazards=np.array([[r[f"halt_probability_{d}"] for d in range(1,4)] for r in draws],dtype=np.float32)
            forced=np.array([[r[f"forced_prediction_{d}"] for d in range(1,5)] for r in draws],dtype=np.int32)
            threshold,_,_=exact_threshold(hazards[~selected])
            depth=hard_depth(hazards[selected],threshold)
            pred=forced[selected][np.arange(len(depth)),depth-1]
            learned={"accuracy":float(np.mean(pred==labels[selected])),"depth":float(depth.mean()),
                     "flops":float(np.mean(forward_flops(w,lengths[selected],depth,True))),"n":int(selected.sum())}
        else:
            pred=np.array([int(r["prediction"]) for r in draws])
            cost=np.array([int(r["matmul_flops"]) for r in draws],dtype=np.int64)
            points.append({"mode":mode,"accuracy":float(np.mean(pred[selected]==labels[selected])),"flops":float(cost[selected].mean())})
    best,_=frontier(points,learned["flops"])
    learned["frontier_gain_pp"]=100*(learned["accuracy"]-best[0]) if best[0] is not None else None
    return learned


def summarize():
    results={}
    for stage,sizes in [("opt",[64]),("long",[896,1600,2880])]:
        with (OUT/f"{stage}-metrics.csv").open() as f:
            baselines=list(csv.DictReader(f))
        for width in sizes:
            rows=[]
            for seed in SEEDS:
                m=json.loads((OUT/f"sensitivity-{stage}-w{width}-adaptive-s{seed}.json").read_text())
                id_metrics=m["splits"]
                n=sum(s["n"] for s in id_metrics)
                cost=sum(s["mean_flops_est"]*s["n"] for s in id_metrics)/n
                d=sum(s["mean_depth"]*s["n"] for s in id_metrics)/n
                acc=sum(s["correct"] for s in id_metrics)/n
                shuffled=sum(np.mean(s["permuted_depth_correct"]) for s in id_metrics)/n
                fixed=[{"mode":p["mode"],"flops":float(p["flops"]),"accuracy":float(p["accuracy"])}
                       for p in baselines if p["group"]=="id" and
                       int(p["seed"])==seed and (stage=="opt" or int(p["width"])==width)
                       and p["mode"].startswith(("untied","fixed"))]
                best,exact=frontier(fixed,cost)
                basepath=OUT/f"{stage}-w{width}-adaptive-s{seed}.json"
                original_thresh=json.loads(basepath.read_text())["threshold"]
                with gzip.open(OUT/f"sensitivity-{stage}-w{width}-adaptive-s{seed}.csv.gz","rt") as handle:
                    same_draws=list(csv.DictReader(handle))
                replay_h=np.array([[r[f"halt_probability_{d}"] for d in range(1,4)] for r in same_draws],dtype=np.float32)
                replay_f=np.array([[r[f"forced_prediction_{d}"] for d in range(1,5)] for r in same_draws],dtype=np.int32)
                replay_y=np.array([int(r["label"]) for r in same_draws])
                coarse_d=hard_depth(replay_h,original_thresh)
                fine_d=hard_depth(replay_h,m["refined_threshold"])
                idx=np.arange(len(coarse_d))
                diff=float(np.mean(replay_f[idx,fine_d-1]==replay_y)-np.mean(replay_f[idx,coarse_d-1]==replay_y))
                half=archived_half(stage,width,seed)
                rows.append({"same_trajectory_cached_coarse_vs_fine_gain_pp":100*diff,
                             "exact_original_checkpoint_archived_half":half,"seed":seed,"depth":d,"dev_depth":m["dev_depth"],"flops":cost,
                             "accuracy":acc,"frontier_gain_pp":100*(acc-best[0]) if best[0] is not None else None,
                             "strict_gain_pp":100*(acc-exact[0]) if exact[0] is not None else None,
                             "shuffle_gain_pp":100*(acc-shuffled),"breakpoints":m["breakpoint_count"],
                             "replay_forced_disagreements":m["replayed_forced_predictions_changed"],
                             "replay_max_hazard_error":m["replayed_max_abs_hazard_change"],
                             "old_hard_depth_vs_replay_cached_disagreements":m["original_policy_vs_replay_cached_depth_changes"],
                             "old_hard_prediction_vs_replay_cached_disagreements":m["original_policy_vs_replay_cached_prediction_changes"],
                             "refined_eval_hard_vs_cached_depth_disagreements":sum(s["cache_depth_disagreement"] for s in id_metrics),
                             "refined_eval_hard_vs_cached_prediction_disagreements":sum(s["cache_prediction_disagreement"] for s in id_metrics)})
            output={"rows":rows}
            for key in ("depth","dev_depth","flops","accuracy","frontier_gain_pp","strict_gain_pp","shuffle_gain_pp","same_trajectory_cached_coarse_vs_fine_gain_pp"):
                nums=[r[key] for r in rows if r[key] is not None]
                output[key]=interval(nums) if len(nums)==3 else None
            for key in ("accuracy","depth","frontier_gain_pp"):
                vals=[r["exact_original_checkpoint_archived_half"][key] for r in rows]
                output["archived_half_"+key]=interval(vals) if None not in vals else None
            results[f"{stage}-{width}"]=output
    (OUT/"sensitivity-summary.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    return results


if __name__=="__main__":
    for name,row in summarize().items():
        print(name,row["depth"],row["frontier_gain_pp"])
