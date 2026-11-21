"""Calculate paired factorial estimates across the predeclared three independent training seeds."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "joint"
AXES = "DGRM"
SEEDS = (41,42,43)
C95 = t.ppf(.975,2)


def interval(x):
    x = np.asarray(x,float)
    assert x.shape==(3,)
    mu = x.mean()
    radius = C95*x.std(ddof=1)/np.sqrt(3)
    return {"mean":float(mu),"lo95":float(mu-radius),"hi95":float(mu+radius),
            **{f"seed{s}":float(v) for s,v in zip(SEEDS,x)}}


def cell(axes):
    return "".join(a for a in AXES if a in axes) or "fixed"


def write_csv(path,rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    with (OUT/"fits.csv").open() as stream:
        allrows=list(csv.DictReader(stream))
    fits={(int(r["seed"]),r["cell"]):r for r in allrows}
    assert len(fits)==len(allrows)==51 and all(r["test_examples"]=="4096" for r in allrows)
    def measure(key, field="accuracy"):
        return np.array([float(fits[s,key][field]) for s in SEEDS])
    configs=[cell(a) for k in range(5) for a in itertools.combinations(AXES,k)]+["hand"]
    means=list()
    for c in configs:
        accuracy=interval(100*measure(c))
        means.append({"cell":c,**{f"accuracy_pct_{k}":v for k,v in accuracy.items()},
            "cross_entropy_nats":measure(c,"cross_entropy").mean(),
            "mean_steps":measure(c,"mean_passes").mean(),
            "active_macs":int(measure(c,"compact_inference_linear_macs")[0]),
            "standard_macs":int(measure(c,"standardized_inference_linear_macs")[0]),
            "trainable_params":int(measure(c,"effective_trainable_parameters")[0]),
            "nominal_train_macs":int(measure(c,"nominal_dense_train_linear_macs")[0]),
            "median_us_example":float(np.median(measure(c,"median_us_per_example_at_batch64")))})
    write_csv(OUT/"cell_means.csv",means)
    contrasts=[]
    def add(name,values):
        contrasts.append({"contrast":name,**interval(100*np.array(values))})
    add("synergy_full_minus_singletons_plus_3fixed",measure("DGRM")-sum(measure(a) for a in AXES)+3*measure("fixed"))
    add("full_minus_fixed",measure("DGRM")-measure("fixed"))
    for c in AXES:
        add("full_minus_"+c,measure("DGRM")-measure(c))
    add("full_minus_hand",measure("DGRM")-measure("hand"))
    for a,b in itertools.combinations(AXES,2):
        other=set(AXES)-{a,b}
        terms=[]
        for k in range(3):
            for context in itertools.combinations(sorted(other),k):
                s=set(context)
                terms.append(measure(cell(s|{a,b}))-measure(cell(s|{a}))-measure(cell(s|{b}))+measure(cell(s)))
        add(f"interaction_{a}{b}_averaged",np.mean(terms,axis=0))
    with (OUT/"interventions.csv").open() as f:
        probe={(int(r["seed"]),r["control"]):r for r in csv.DictReader(f)}
    for variant in ["tuple",*AXES]:
        add("full_minus_permuted_"+variant,[float(probe[s,variant]["natural_minus_intervention"]) for s in SEEDS])
    for teacher in (1,3):
        add(f"forced3_minus_forced1_given_teacher{teacher}",[float(probe[s,f"force3_teacher{teacher}"]["accuracy"])-
            float(probe[s,f"force1_teacher{teacher}"]["accuracy"]) for s in SEEDS])
    write_csv(OUT/"contrasts_pp.csv",contrasts)
    diag=[{"metric":key,**interval(measure("DGRM",key))} for key in
            [a+suffix for a in AXES for suffix in ("_teacher_agreement","_activation_rate")]+["R_permutation_invariant_agreement"]]
    write_csv(OUT/"policy_diagnostics.csv",diag)
    frontier=[]
    for row in means:
        if row["cell"]=="hand":
            continue
        x,y=row["active_macs"],row["accuracy_pct_mean"]
        if not any(r["cell"]!="hand" and r["active_macs"]<=x and r["accuracy_pct_mean"]>=y and
                  (r["active_macs"]<x or r["accuracy_pct_mean"]>y) for r in means):
            frontier.append(row["cell"])
    summaries={"frontier_learned_cells_mean_accuracy_vs_compact_macs":frontier,
        "aggregate_train_seconds":sum(float(r["train_seconds"]) for r in allrows),
        "total_nominal_train_linear_macs":sum(int(r["nominal_dense_train_linear_macs"]) for r in allrows),
        "trained_examples_all_models":sum(int(r["train_examples"]) for r in allrows),
        "test_examples_common":4096,"num_fits":len(allrows)}
    (OUT/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    plot(means,contrasts)


def plot(means,contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.5,4.9),gridspec_kw={"width_ratios":[1.27,1]})
    for row in means:
        x=row["active_macs"]
        y=row["accuracy_pct_mean"]
        hand=row["cell"]=="hand"
        special = row["cell"] in ("fixed","M","GM","DGRM","hand")
        ax.plot([x]*3,[row[f"accuracy_pct_seed{s}"] for s in SEEDS],".",color="#858585",alpha=.35)
        ax.scatter(x,y,marker="s" if hand else "o",s=34 if special else 18,
                   color="#a04621" if hand else ("#025065" if special else "#768789"),zorder=4)
        if special:
            ax.annotate(row["cell"],(x,y),xytext=(2,5),textcoords="offset points",fontsize=8)
    ax.set_xlabel("Active linear MACs per example")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("a    All controller combinations",loc="left",fontsize=11)
    items=[r for r in contrasts if r["contrast"].startswith("interaction_")]
    for i,r in enumerate(items):
        bx.plot([r["lo95"],r["hi95"]],[i,i],color="#025065",lw=1.8)
        bx.scatter(r["mean"],i,color="#025065",s=29)
    bx.axvline(0,color="#555",lw=.8,ls="--",zorder=0)
    bx.set_yticks(range(len(items)),[r["contrast"].split("_")[1] for r in items])
    bx.invert_yaxis()
    bx.set_xlabel("Accuracy interaction (percentage points)")
    bx.set_title("b    Mean pairwise interactions",loc="left",fontsize=11)
    for a in (ax,bx):
        a.spines[["top","right"]].set_visible(False)
        a.grid(axis="y",alpha=.13)
    fig.tight_layout()
    fig.savefig(OUT/"factorial.png",dpi=180,bbox_inches="tight")
    fig.savefig(OUT/"factorial.pdf",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
