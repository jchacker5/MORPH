"""Typeset the archived paired results, showing both initial and extended regimes."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULT=HERE.parent/"results/reasoning"
def read(fname): return json.loads((RESULT/fname).read_text())
def pp(x): return f"{x:+.2f}".replace("-","$-$")
def interval(a): return f"{pp(a['mean'])} [{pp(a['lo'])}, {pp(a['hi'])}]"
def row(cells): return " & ".join(map(str,cells)) + r"\\"
def pct(mode,group): return f"{mode[group]['accuracy']['mean']*100:.2f}"
def label(k): return {"id":"ID, 32--128","id64":"ID, 64","length256":"Length 256","length512":"Length 512","length1024":"Length 1,024","composition64":"4--6 ops, 64"}[k]
def main_section():
    p=read("main-summary.json")["pilot"]
    j=read("opt-summary.json")["pilot"]
    m=j['model']; a=m['adaptive']
    q=read("usability-summary.json")
    c=j['adaptive_comparison']
    lines=[r"\section*{3\quad Depth, interventions and extrapolation}",
        r"\topic{E1.} With 120 updates the 50,563-parameter adaptive model averaged "+pct(p['model']['adaptive'],'id')+
        r"\% in distribution. Its difference from the best fixed policy within budget was "+
        interval(p['adaptive_comparison']['id']['frontier_gain_pp'])+
        r" pp. The chance-level pilot motivated a separately frozen follow-up. All seven models were reinitialized, used new training and evaluation draws, and received 2,400 updates of 24 examples with learning-rate warmup, decay and clipping.",
        r"\begin{center}\includegraphics[width=.97\linewidth]{figures/reasoning.pdf}\end{center}",
        r"\captext{Figure 1. Mean of three paired seeds after longer training. Numbers mark fixed recurrent depths; faint points at left are individual adaptive and untied seeds. ID lengths use 1--3 instructions.}",
        r"\begin{center}\small\begin{tabular}{lrrrr}\toprule",
        row(["Test condition","MORPH, \\%","Fixed two, \\%","Passes","Gain vs. fixed frontier, pp"])+r"\midrule"]
    for key in ["id","id64","length256","length512","length1024","composition64"]:
        lines.append(row([label(key),pct(a,key),pct(m['fixed2'],key),f"{a[key]['depth']['mean']:.2f}",interval(c[key]['frontier_gain_pp'])]))
    lines += [r"\bottomrule\end{tabular}\end{center}",
        r"\topic{E6--7.} Reassigning the observed exits across inputs of the same length changed accuracy. The original-minus-shuffled difference on pooled ID examples was "+interval(a['id']['shuffle_gain_pp'])+
        r" pp. Counts, lengths and individual hard-depth interventions are archived for every seed. Comparing the 64-token rows isolates instruction count; the longer rows change only length.",
        r"\topic{Qualification.} In a separate 8,000-update, seed-17 run on 14,293,248 token presentations, adaptive scored 1,087/1,152 (94.36\%) and fixed two scored 1,136/1,152 (98.61\%). Only fixed two passed the prespecified ID criterion. Their 1,024-token accuracies were 56.25\% and 65.63\%, respectively. Neither passed the shift criterion. Fixed two consumed 10.54 estimated training TFLOPs against 21.08 for adaptive."]
    (HERE/"reasoning-results.tex").write_text("\n".join(lines).replace("\n\n"+" & ","\n"+" & ")+"\n")

def scale_section():
    p=read("scale-summary.json");j=read("long-summary.json")
    lines=[r"\section*{6\quad Does a depth advantage persist at scale?}",
        r"\topic{E8.} We widened the shared Boolean-program block to 896, 1,600 and 2,880 hidden dimensions. Every parameter belongs to an active computation or readout. The first ladder used 16 updates and 1,024 input token presentations per fit, with three paired seeds for fixed four and adaptive. It was a feasibility test. Training-time profiling motivated a second, separately registered 45-fit ladder with five independently trained models, three sizes and three seeds. Each fit received 160 updates, 1,280 examples and 40,960 input token presentations at length 32. The test used a new 256-example draw. The parameter count and training cost change together across sizes.",
        r"\begin{center}\small\begin{tabular}{lrrrrrr}\toprule",
        row(["Params","MORPH, \\%","Fixed 1","Fixed 2","Fixed 3","Fixed 4","Train TFLOPs"])+r"\midrule"]
    for width,r in j.items():
        m=r['model'];a=m['adaptive']['id']
        lines.append(row([f"{a['params']['mean']/1e6:.2f}M",pct(m['adaptive'],'id'),*[pct(m[f'fixed{i}'],'id') for i in range(1,5)],f"{a['train_flops']['mean']/1e12:.2f}"]))
    lines += [r"\bottomrule\end{tabular}\end{center}",
        r"\captext{Accuracy values are percentages on the controlled synthetic task. The last column is adaptive training arithmetic per fit. At each width the fixed-four model closely matches both the adaptive model's trainable parameters and its training arithmetic. Fixed one through four locate its inference frontier.}",
        r"\begin{center}\includegraphics[width=.93\linewidth]{figures/scaling.pdf}\end{center}",
        r"\captext{Figure 2. Adaptive accuracy minus the best feasible fixed model or input-independent fixed mixture. Points are paired training seeds. Teal lines indicate means; zero indicates parity.}",
        r"\topic{Primary comparison.} The respective gains in percentage points at the three sizes were "+", ".join(interval(v['adaptive_comparison']['id']['frontier_gain_pp']) for v in j.values())+
        r". On the original 16-update ladder adaptive accuracy was "+", ".join(pct(v['model']['adaptive'],'id')+r"\%" for v in p.values())+
        r". Different test drawings and training schedules make the two ladders unsuitable for estimating a causal effect of training duration.",
        r"\input{scale-sensitivity.tex}"]
    (HERE/"scale-results.tex").write_text("\n".join(lines)+"\n")
if __name__=="__main__":
    main_section(); scale_section()
