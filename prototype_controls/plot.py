"""Plot measured 30M prediction quality against hard-inference matrix FLOPs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/prototype_controls"


def main():
    with (OUT/"all_scores.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(9.8,4.2),sharey=True)
    colors={"512":"#075e65","2048":"#a65230"}
    for ax,subset,title in zip(axes,("plays","sonnets"),("Held-out play text","External sonnets")):
        for steps in ("512","2048"):
            # The external graph uses the single seed shared with adaptive weights.
            selected=[r for r in rows if r["steps"]==steps and r["subset"]==subset and
                      (subset=="plays" or r["seed"]=="11")]
            base=[]
            for name in ("fixed1","fixed2","fixed4","adaptive"):
                cell=[r for r in selected if r["model"]==name]
                if not cell: continue
                x=np.array([float(r["estimated_inference_flops_per_window"])/1e9 for r in cell])
                y=np.array([float(r["bpb"]) for r in cell])
                marker="D" if name=="adaptive" else "o"
                ax.scatter(x,y,color=colors[steps],marker=marker,alpha=.22,s=17,zorder=3)
                ax.scatter(x.mean(),y.mean(),color=colors[steps],marker=marker,s=45,edgecolors="white",linewidths=.6,zorder=5)
                if name!="adaptive": base.append((x.mean(),y.mean()))
            if base:
                xx,yy=zip(*sorted(base))
                ax.plot(xx,yy,color=colors[steps],lw=.9,alpha=.6,zorder=2)
        ax.set_title(title,loc="left",fontsize=11)
        ax.grid(axis="y",color="#ced3d4",lw=.5)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlabel("Inference GFLOPs / 64-byte window")
    axes[0].set_ylabel("Held-out bits per byte (lower is better)")
    from matplotlib.lines import Line2D
    legend=[Line2D([],[],marker=m,color="#333",lw=0,label=label) for m,label in [("o","Independent fixed depth"),("D","Adaptive hard exit")]]
    legend += [Line2D([],[],color=v,label=k+" updates") for k,v in colors.items()]
    fig.legend(handles=legend,loc="lower center",ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=[0,.08,1,1])
    fig.savefig(OUT/"shallow-controls.pdf",bbox_inches="tight")
    fig.savefig(OUT/"shallow-controls.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
