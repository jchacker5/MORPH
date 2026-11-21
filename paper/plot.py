"""Draw fixed-compute and shift plots from the archived seed summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parent/"results/reasoning"
OUT=HERE/"figures"
TEAL="#087d73"; PLUM="#884453"; DARK="#273e4a"; GREY="#65777e"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#a6b4b4",
                     "axes.labelcolor":DARK,"text.color":DARK,"xtick.color":GREY,"ytick.color":GREY,"pdf.fonttype":42})

def save(fig,name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",pad_inches=.05)
    fig.savefig(OUT/f"{name}.png",dpi=170,bbox_inches="tight",pad_inches=.05)
    plt.close(fig)

def ood():
    j=json.loads((RESULTS/"opt-summary.json").read_text())["pilot"]
    m=j["model"]
    fig,axes=plt.subplots(1,2,figsize=(7.45,2.45),layout="constrained")
    ax=axes[0]
    order=["fixed1","fixed2","fixed3","fixed4"]
    xs=[m[k]["id"]["flops"]["mean"]/1e6 for k in order]
    ys=[m[k]["id"]["accuracy"]["mean"]*100 for k in order]
    ax.plot(xs,ys,color=GREY,lw=1.3,marker="o",label="Fixed recurrent depths")
    for i,(x,y) in enumerate(zip(xs,ys)): ax.annotate(str(i+1),(x,y),xytext=(4,4),textcoords="offset points",color=GREY)
    for k,label,color,mark in [("adaptive","MORPH",TEAL,"D"),("untied4","Untied four",PLUM,"s"),("random","Random depth","#926722","^")]:
        a=m[k]["id"]
        ax.scatter(np.asarray(a["flops"]["values"])/1e6,np.asarray(a["accuracy"]["values"])*100,s=11,color=color,alpha=.40,zorder=3)
        ax.scatter(a["flops"]["mean"]/1e6,a["accuracy"]["mean"]*100,color=color,marker=mark,s=35,label=label,zorder=4)
    ax.set(title="A   In-distribution reasoning",xlabel="Estimated MFLOPs per input",ylabel="Exact accuracy (%)")
    ax.legend(frameon=False,fontsize=6.7,loc="lower right"); ax.grid(axis="y",alpha=.18)
    ax=axes[1]
    keys=["id","length256","length512","length1024"]
    for k,label,color,mark in [("adaptive","MORPH",TEAL,"D"),("fixed2","Fixed two",DARK,"o"),("fixed4","Fixed four",PLUM,"s")]:
        acc=[m[k][s]["accuracy"]["mean"]*100 for s in keys]
        ax.plot(range(4),acc,marker=mark,color=color,markersize=4,lw=1.3,label=label)
    ax.axhline(50,ls=":",color="#9f9f9f",linewidth=1)
    ax.set(title="B   Length shift, 1–3 instructions",ylabel="Exact accuracy (%)",xlabel="Input length (tokens)",xticks=range(4),xticklabels=["32–128","256","512","1,024"])
    ax.legend(frameon=False,fontsize=6.7); ax.grid(axis="y",alpha=.18)
    save(fig,"reasoning")

def scaling():
    obj=json.loads((RESULTS/"long-summary.json").read_text())
    fig,ax=plt.subplots(figsize=(6.9,2.05),layout="constrained")
    for idx,(width,v) in enumerate(obj.items()):
        info=v["adaptive_comparison"]["id"]["frontier_gain_pp"]
        vals=info["values"]
        ax.plot([idx-.12,idx,idx+.12],vals,ls="none",marker="o",color=GREY,alpha=.67,markersize=4.8)
        ax.plot([idx-.25,idx+.25],[info["mean"]]*2,lw=2.0,color=TEAL,label="Mean" if idx==0 else None)
    ax.set_xticks(range(3),["9.65M","30.76M","99.60M"])
    ax.set(xlabel="Parameters in the adaptive Boolean-program model",ylabel="MORPH minus fixed frontier (pp)")
    ax.axhline(0,ls="--",lw=1,color=PLUM); ax.grid(axis="y",alpha=.18)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    save(fig,"scaling")

if __name__=="__main__":
    ood(); scaling()
