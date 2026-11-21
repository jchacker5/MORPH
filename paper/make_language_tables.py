"""Regenerate tabular text and cost-envelope contrasts for the manuscript."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
HERE=Path(__file__).resolve().parent
R=HERE.parent/"results"
SEEDS=(11,23,37)
def load(p):return json.loads(p.read_text())
def row(cells):return " & ".join(map(str,cells))+r"\\"
def f(x):return f"{x:.3f}"
def interval(vals):
    x=np.asarray(vals); avg=x.mean(); delta=t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/np.sqrt(len(x))
    return f"{avg:+.3f} [{avg-delta:+.3f}, {avg+delta:+.3f}]"
def long():
    models={m:[load(R/f"lm_followup/long-{m}-{s}.json") for s in SEEDS] for m in ("1","2","4","adaptive")}
    parts=[r"\begin{center}\begin{tabular}{lrrr}\toprule",
        row(["4,096 updates, 118K","Plays BPB","Sonnets BPB","Sonnets passes"])+r"\midrule"]
    for mode,fits in models.items():
        parts.append(row([("MORPH" if mode=="adaptive" else "Fixed "+mode),f(np.mean([j['repeat_test']['bpb'] for j in fits])),
            f(np.mean([j['sonnets']['bpb'] for j in fits])),f"{np.mean([j['sonnets']['mean_depth'] for j in fits]):.2f}"]))
    parts += [r"\bottomrule\end{tabular}\end{center}",
        "The prespecified longer-run adaptive-minus-fixed-two contrast on the sonnets was "+interval([a['sonnets']['bpb']-b['sonnets']['bpb'] for a,b in zip(models['adaptive'],models['2'])])+
        " BPB. Each seed saw the same training windows across architectures. The play corpus had already been evaluated in the original comparison."]
    (HERE/"long-text-rows.tex").write_text("\n".join(parts)+"\n")

def frontier(cases,budget):
    feasible=[(q,name) for (name,q,c) in cases if c<=budget]
    for (na,qa,ca),(nb,qb,cb) in itertools.permutations(cases,2):
        if ca<budget<cb:
            frac=(budget-ca)/(cb-ca)
            feasible.append(((1-frac)*qa+frac*qb,f"{na}+{nb}@{frac:.5f}"))
    return min(feasible)

def wide():
    with (R/"prototype_controls/all_scores.csv").open() as fp: scores=list(csv.DictReader(fp))
    def select(m,step,subset,seed=None):
        return [x for x in scores if x['model']==m and int(x['steps'])==step and x['subset']==subset and (seed is None or int(x['seed'])==seed)]
    parts=[r"\begin{center}\small\begin{tabular}{lrrrr}\toprule",
        row(["30M model","Plays, 512","Passes, 512","Plays, 2,048","Sonnets, 2,048"])+r"\midrule"]
    for m in ['fixed1','fixed2','fixed4','adaptive']:
        multi=select(m,512,'plays');end=select(m,2048,'plays')[0]
        external=select(m,2048,'sonnets')
        parts.append(row([("MORPH" if m=="adaptive" else "Fixed "+m[-1]),f(np.mean([float(r['bpb']) for r in multi])),f"{np.mean([float(r['hard_depth']) for r in multi]):.2f}",f(float(end['bpb'])),f(float(external[0]['bpb'])) if external else "---"]))
    parts += [r"\bottomrule\end{tabular}\end{center}",r"\captext{Lower BPB is better. The 512-update plays results average three independent seeds. The 2,048-update endpoint uses seed 11 alone. The fixed-four Sonnets value was not measured.}"]
    (HERE/"wide-rows.tex").write_text("\n".join(parts)+"\n")
    effects=[]
    for s in SEEDS:
        a=select('adaptive',512,'plays',s)[0]
        pts=[]
        for m in ['fixed1','fixed2','fixed4']:
            j=select(m,512,'plays',s)[0]
            pts.append((m,float(j['bpb']),float(j['estimated_inference_flops_per_window'])))
        fixed,policy=frontier(pts,float(a['estimated_inference_flops_per_window']))
        effects.append(dict(seed=s,delta_bpb=float(a['bpb'])-fixed,fixed_bpb=fixed,policy=policy))
    (R/"prototype_controls/frontier.json").write_text(json.dumps(effects,indent=2)+"\n")
    a=select('adaptive',2048,'plays')[0]
    comparison=interval([r['delta_bpb'] for r in effects])
    text=(r"\topic{At the observed budget.} On plays, the 512-update adaptive-minus-best-feasible-fixed BPB difference was "+
          comparison+" across paired seeds. The descriptive envelope selects a fixed policy or mix at or below the measured cost on the same test set. "+
          "At 2,048 updates the exported adaptive model scored "+f(float(a['bpb']))+" BPB at "+
          f"{float(a['hard_depth']):.2f} mean passes. The 118K capstone saw substantially more data than this model. Size and training exposure are confounded in cross-size language comparisons.")
    (HERE/"wide-verdict.tex").write_text(text+"\n")

if __name__=="__main__":long();wide()
