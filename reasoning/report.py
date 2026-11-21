"""Regenerate the reasoning supplement and figure from the verified run outputs."""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT
from analyze_sensitivity import summarize as summarize_sensitivity

MODES=["fixed1","fixed2","fixed3","fixed4","untied4","random","adaptive"]
LABEL={"fixed1":"Shared fixed 1","fixed2":"Shared fixed 2","fixed3":"Shared fixed 3",
       "fixed4":"Shared fixed 4","untied4":"Untied fixed 4","random":"Random depth","adaptive":"MORPH"}
GROUPS=["id","id64","length256","length512","length1024","composition64"]


def print_value(v, factor=1):
    if v is None:
        return "unavailable"
    return f"{v['mean']*factor:.2f} [{v['lo']*factor:.2f}, {v['hi']*factor:.2f}]"


def accuracy(record):
    return f"{100*record['accuracy']['mean']:.2f}"


def wilson(correct,n):
    z=1.95996398454
    p=correct/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def qualification():
    out={}
    for mode in ("adaptive","fixed2"):
        rec=json.loads((OUT/f"usable-w64-{mode}-s17.json").read_text())
        splits=rec["splits"]
        id_set=[s for s in splits if s["name"].startswith("id")]
        n=sum(s["n"] for s in id_set)
        k=sum(s["correct"] for s in id_set)
        w=wilson(k,n)
        valid=k/n>=.95 and w[0]>=.93
        shifts={s["name"]:s["accuracy"] for s in splits if s not in id_set}
        out[mode]={"id_correct":k,"id_n":n,"id_accuracy":k/n,"wilson95":list(w),"id_qualified":valid,
                   "shift_qualified":valid and min(shifts.values())>=.9,"shifts":shifts,"threshold":rec["threshold"],
                   "train_tokens":rec["train_tokens"],"train_flops_est":rec["train_flops_est"]}
    (OUT/"usability-summary.json").write_text(json.dumps(out,indent=2)+"\n")
    return out


def figure(opt, scale):
    fig,axes=plt.subplots(1,3,figsize=(13.3,3.9),layout="constrained")
    m=opt["model"]
    ax=axes[0]
    fixed=[m[f"fixed{i}"]["id"] for i in range(1,5)]
    ax.plot([s["flops"]["mean"]/1e6 for s in fixed],[s["accuracy"]["mean"]*100 for s in fixed],
            color="#31998d",marker="o",label="Shared fixed")
    for mode,marker,color in [("adaptive","D","#ba4a2c"),("random","x","#777777"),("untied4","s","#263c78")]:
        row=m[mode]["id"]
        ax.scatter([row["flops"]["mean"]/1e6],[row["accuracy"]["mean"]*100],marker=marker,s=48,color=color,label=LABEL[mode],zorder=3)
    ax.set_xlabel("Estimated MFLOPs per sequence"); ax.set_ylabel("Exact accuracy (%)"); ax.set_title("A. Compute and accuracy",loc="left")
    ax.legend(fontsize=8,frameon=False)
    ax=axes[1]
    groups=["id64","length256","length512","length1024","composition64"]
    for mode,clr,sym in [("fixed4","#31998d","o"),("adaptive","#ba4a2c","D")]:
        points=[m[mode][g]["accuracy"]["mean"]*100 for g in groups]
        ax.plot(range(4),points[:4],marker=sym,color=clr,label=LABEL[mode])
        ax.scatter([4],[points[4]],marker=sym,color=clr)
    ax.axvline(3.5,color=".75",lw=1,ls=":")
    ax.set_xticks(range(5),["64","256","512","1024","64*"]); ax.set_xlabel("Length (*unseen instruction counts)")
    ax.set_ylabel("Exact accuracy (%)");ax.set_title("B. Frozen policy under shift",loc="left");ax.legend(fontsize=8,frameon=False)
    ax=axes[2]
    widths=("896","1600","2880")
    gains=[scale[w]["adaptive_comparison"]["id"]["frontier_gain_pp"] for w in widths]
    if all(v is not None for v in gains):
        y=[v["mean"] for v in gains]
        ax.errorbar(range(3),y,yerr=[np.array(y)-[v["lo"] for v in gains],[v["hi"] for v in gains]-np.array(y)],
                    color="#ba4a2c",fmt="D",capsize=4)
    ax.axhline(0,ls="--",lw=1,color=".45")
    ax.set_xticks(range(3),["9.65","30.76","99.60"]);ax.set_xlabel("Active model size (M parameters)")
    ax.set_ylabel("MORPH minus fixed frontier (pp)");ax.set_title("C. Independently trained widths",loc="left")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.16)
    for ext in ("png","pdf"):
        fig.savefig(OUT / f"reasoning-frontiers.{ext}",dpi=185)
    plt.close(fig)


def main():
    main=json.loads((OUT/"main-summary.json").read_text())["pilot"]
    opt=json.loads((OUT/"opt-summary.json").read_text())["pilot"]
    smoke=json.loads((OUT/"scale-summary.json").read_text())
    scale=json.loads((OUT/"long-summary.json").read_text())
    q=qualification()
    document=["# MORPH Boolean reasoning study", "", "Joseph Defendre", "",
              "Experiments 1, 6, 7, and 8. Each primary comparison uses three paired training seeds and identical test draws within a study. Intervals are descriptive two-sided 95% Student-t intervals for the mean over seeds, conditional on the locked test sample. Every quality percentage is exact final-bit accuracy.","",
              "## Matched-compute reasoning results","",
              "The initial study used 120 updates per fit. The registered follow-up used 2,400 updates on fresh streams after the pilot suggested undertraining. Each row in the following table represents three independent fits in each study. Costs use theoretical forward matrix products. Data generation and optimizer steps were matched within each comparison.","",
              "| Inference policy | Parameters | Pilot accuracy (%) | Longer accuracy (%) | Longer MFLOPs/example | Longer passes |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in MODES:
        p=main["model"][name]["id"]; o=opt["model"][name]["id"]
        document.append(f"| {LABEL[name]} | {int(o['params']['mean']):,} | {accuracy(p)} | {accuracy(o)} | {o['flops']['mean']/1e6:.2f} | {o['depth']['mean']:.2f} |")
    for label,study in [("Pilot",main),("Longer study",opt)]:
        compare=study["adaptive_comparison"]["id"]
        d=study["model"]["adaptive"]["id"]
        document += ["",f"{label}. MORPH minus the best tested fixed policy or fixed mixture within its FLOP budget was {print_value(compare['frontier_gain_pp'])} percentage points. The strict equal-cost mixture gave {print_value(compare['exact_mix_gain_pp'])} points. The fixed envelope is chosen from the observed test means for each seed and is therefore an optimistic comparator. The gain over fixed-four under closely matched training cost was {print_value(compare['fixed4_gain_pp'])} points; relative training-cost difference was {100*compare['training_cost_relative_gap']['mean']:.5f}%.",
                    "", f"The uncalibrated 0.5 policy used {d['raw_depth']['mean']:.2f} mean passes. The development-calibrated policy used {d['depth']['mean']:.2f}. Compared with 32 length-stratified exit permutations it gained {print_value(d['shuffle_gain_pp'])} points. Seed-wise instruction-count/depth Spearman coefficients were {d['depth_count_rho']}. Forced depths 1–4 gave {', '.join(f'{d[f'forced{j}_accuracy']['mean']*100:.2f}%' for j in range(1,5))}. The archived rows include actual hard exits, forced predictions, and controller probabilities."]
    document += ["","## Generalization after longer training","",
                 "| Split | MORPH (%) | Shared fixed four (%) | MORPH passes | MORPH vs fixed frontier (pp, 95% seed interval) |",
                 "|---|---:|---:|---:|---:|"]
    for g in GROUPS:
        a=opt["model"]["adaptive"][g]; f=opt["model"]["fixed4"][g]
        document.append(f"| {g} | {accuracy(a)} | {accuracy(f)} | {a['depth']['mean']:.2f} | {print_value(opt['adaptive_comparison'][g]['frontier_gain_pp'])} |")
    document += ["","The ID pool has 576 examples, 192 at each of 32, 64 and 128 tokens. Shift rows contain 96 examples each. The composition row has 4–6 instructions at length 64, while id64 has 1–3 instructions at that same length. Negative generalization results must be interpreted within this Boolean execution task.","",
                 "## Scaling synthetic reasoning","",
                 "| MORPH parameters | Pilot adaptive/fixed-four (%) | Longer adaptive (%) | Longer fixed 1/2/3/4 (%) | Adaptive gain vs fixed envelope (pp, 95% seed interval) | Training TFLOPs/fit, MORPH |",
                 "|---:|---:|---:|---:|---:|---:|"]
    for width,res in scale.items():
        a=res["model"]["adaptive"]["id"]; base=smoke[width]["model"]
        vals=[accuracy(res["model"][f"fixed{v}"]["id"]) for v in range(1,5)]
        document.append(f"| {int(a['params']['mean']):,} | {accuracy(base['adaptive']['id'])}/{accuracy(base['fixed4']['id'])} | {accuracy(a)} | {' / '.join(vals)} | {print_value(res['adaptive_comparison']['id']['frontier_gain_pp'])} | {a['train_flops']['mean']/1e12:.3f} |")
    document += ["","The pilot had 16 updates and 1,024 training tokens per fit. The longer replication had 160 updates and 40,960 training tokens per fit across all 45 models. Its data budget is too small to infer an asymptotic scaling law. Width, active parameters and FLOPs change together across sizes. Within each size the trained policies saw the same input tokens. Each adaptive model trains all four exits and is compared independently with a parameter- and training-FLOP-matched shared fixed-four model and an inference-cost frontier.","",
                 "## Engineering qualification","",
                 "Two fresh seed-17 models trained for 8,000 updates on identical data, then ran on a separate qualification split. Pooled ID qualification requires 95% observed accuracy and a 93% Wilson lower bound. Shift qualification also requires at least 90% in each shift.","",
                 "| Checkpoint | Correct / ID examples | ID accuracy (Wilson 95%) | ID qualified | Shift-qualified | Accuracy at lengths 256 / 512 / 1,024 / composition (%) |",
                 "|---|---:|---:|---|---|---:|"]
    for mode,row in q.items():
        d=row["shifts"]; a=row["id_accuracy"]; lo,hi=row["wilson95"]
        document.append(f"| {LABEL[mode]} | {row['id_correct']} / {row['id_n']} | {100*a:.2f} [{100*lo:.2f}, {100*hi:.2f}] | {row['id_qualified']} | {row['shift_qualified']} | {' / '.join(f'{100*x:.2f}' for x in d.values())} |")
    document += ["","The four evaluation shifts are assessed per checkpoint. These are runnable binary program classifiers over the generated task grammar. General-purpose or conversational performance was outside this reasoning benchmark.","",
                 "## Reproduce","",
                 "Run the following from the standalone project root after installing the locked dependencies.","",
                 "```sh",".venv/bin/python reasoning/suite.py all",".venv/bin/python reasoning/extended.py",".venv/bin/python reasoning/optimize.py",".venv/bin/python reasoning/qualify.py",".venv/bin/python reasoning/verify.py --full",
                 "for study in main scale opt long; do .venv/bin/python reasoning/summarize.py \"$study\"; done",
                 ".venv/bin/python reasoning/report.py",".venv/bin/python reasoning/predict.py 1 not zero not --checkpoint adaptive --length 64","```","",
                 "The per-example compressed CSV files, raw per-run JSON, generated metrics tables, train-step times and theoretical FLOP estimates accompany this supplement. Calibrated hard exits physically compact the batch; the cached full-depth analysis permits interventions. Machine timings share the local accelerator with other work. MLX peak allocation is process-scoped. Data and optimizer schedules differ between the pilot and registered follow-ups; their differences are descriptive, not an isolated treatment effect.",""]
    sensitivity=summarize_sensitivity()
    document += ["## Exploratory halt-threshold sensitivity","",
                 "The prespecified primary search uses a 0.001-spaced grid. A post hoc search over every development hazard breakpoint isolated whether this spacing masked input dependence. Both policies use the same development data, 2.5-pass objective and lowest-threshold tie rule. This sensitivity reuses the original held-out examples.","",
                 "| Width | Mean calibrated test passes | Refined accuracy (%) | Gain vs fixed frontier (pp, 95% seed interval) | Shuffle gain (pp) |",
                 "|---|---:|---:|---:|---:|"]
    for name,val in sensitivity.items():
        document.append(f"| {name} | {val['depth']['mean']:.2f} | {100*val['accuracy']['mean']:.2f} | {print_value(val['frontier_gain_pp'])} | {val['shuffle_gain_pp']['mean']:.2f} |")
    document += ["","The companion sensitivity JSON quantifies replay and compact-versus-cached disagreements. When original scaled weights had been discarded, replay under the recorded seed sometimes produced different predictions. Within-replay comparisons hold the weights fixed, and a second cached check recalibrates using the first half of archived original test hazards at each length and scores the other half against original fixed baselines. Both are post hoc checks on reused test data.",""]
    document += ["| Width | Original-weight second-half passes | Second-half accuracy (%) | Second-half gain vs fixed envelope (pp) | Within-replay coarse-to-refined accuracy change (pp) |",
                 "|---|---:|---:|---:|---:|"]
    for name,val in sensitivity.items():
        document.append(f"| {name} | {val['archived_half_depth']['mean']:.2f} | {100*val['archived_half_accuracy']['mean']:.2f} | {print_value(val['archived_half_frontier_gain_pp'])} | {val['same_trajectory_cached_coarse_vs_fine_gain_pp']['mean']:.2f} |")
    document += ["","All scientific study rows above use 21 original pilot fits, 18 original scale fits, 21 extended small fits, 45 extended scale fits and two single-seed engineering qualifications. Profiling and sensitivity replays are separate.",""]
    (OUT/"findings.md").write_text("\n".join(document))
    figure(opt,scale)


if __name__=="__main__":
    main()
