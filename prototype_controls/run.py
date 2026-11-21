"""Run or resume the predeclared shallow prototype controls, one model per process."""
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"lm"),str(ROOT)]
from prepare import WINDOW, PREFIX, evaluation_windows, split_data, windows_at
from prototype import B, WideDecoder, cost_for, evaluate
from sonnets import eval_windows

OUT=ROOT/"results/prototype_controls"
PRIMARY=ROOT/"results/lm/prototype"
SEEDS=(11,23,37)


def pin():
    for filename in ("dependencies.sha256","protocol.sha256"):
        for row in (OUT/filename).read_text().splitlines():
            expected,path=row.split(None,1)
            # Use the original registered bytes of the external loader preserved
            # locally after its upstream working copy was reformatted.
            source=ROOT/("prototype_controls/sonnets.py" if path=="lm_followup/data.py" else path)
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError("Registered dependency changed: "+path)


def save(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def datasets():
    return {"plays":evaluation_windows("test"),"sonnets":eval_windows()}


def train(depth,seed):
    name=str(depth)
    mx.random.seed(seed)
    model=WideDecoder(False)
    opt=optim.AdamW(learning_rate=.0003,weight_decay=.01)
    opt.init(model.trainable_parameters())
    gradient=nn.value_and_grad(model,model.objective)
    @partial(mx.compile,inputs=[model.state,opt.state],outputs=[model.state,opt.state])
    def update(rows):
        loss,grad=gradient(rows,name,depth,0.)
        opt.update(model,grad)
        return loss
    mx.eval(model.parameters(),opt.state)
    data=split_data("train")
    tests=datasets()
    rng=np.random.default_rng(seed)
    # The source pipeline's block/window selection is identical in both fixed and adaptive fits.
    nsteps=2048 if seed==11 else 512
    count=sum(v.size for _,v in tree_flatten(model.parameters()))
    tick=time.perf_counter()
    elapsed=0.
    mx.reset_peak_memory()
    for step in range(1,nsteps+1):
        starts=rng.integers(0,len(data)-WINDOW,size=B)
        loss=update(mx.array(windows_at(data,starts)))
        mx.eval(loss,model.parameters(),opt.state)
        if step%256==0:
            print(f"depth={depth} seed={seed} step={step} elapsed={elapsed+time.perf_counter()-tick:.1f}s",flush=True)
        if step in (512,2048):
            elapsed+=time.perf_counter()-tick
            row={"depth":depth,"seed":seed,"steps":step,"lr":.0003,"batch":B,"parameters":count,
                 "input_bytes":step*B*WINDOW,"scored_train_bytes":step*B*(WINDOW-PREFIX),
                 "train_flops":int(step*B*cost_for(depth,False,True)),"last_train_loss":float(loss.item()),
                 "training_peak_interval_mlx_bytes":mx.get_peak_memory(),"training_seconds_including_compile":elapsed}
            row['scores']={s:evaluate(model,name,examples) for s,examples in tests.items()}
            for score in row['scores'].values():
                assert score['mean_depth']==depth
            save(OUT/f"fixed{depth}-{seed}-{step}.json",row)
            print(f"saved fixed{depth}-{seed}-{step}",flush=True)
            mx.reset_peak_memory()
            tick=time.perf_counter()


def adaptive(step):
    seed=11
    name=f"adaptive-{seed}-{step}"
    path=PRIMARY/f"{name}.safetensors"
    saved=json.loads((PRIMARY/f"{name}.json").read_text())
    model=WideDecoder(True)
    model.load_weights(str(path))
    model.eval()
    scores={s:evaluate(model,"adaptive",examples) for s,examples in datasets().items()}
    if abs(scores["plays"]["bpb"]-saved["test"]["bpb"])>1e-5 or scores["plays"]["histogram"]!=saved["test"]["histogram"]:
        raise ValueError("Replayed original-test BPB or exit histogram differs.")
    row={"source_checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"seed":seed,"steps":step,
         "train_flops":saved["train_flops"],"input_bytes":saved["input_bytes"],"parameters":saved["parameters"],
         "scores":scores}
    save(OUT/f"reloaded-{name}.json",row)
    print("reloaded",name,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth",type=int,choices=[1,2])
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--adaptive",type=int,choices=[512,2048])
    args=parser.parse_args()
    pin()
    if args.depth is not None:
        if args.seed is None: parser.error("Supply a training seed")
        return train(args.depth,args.seed)
    if args.adaptive is not None: return adaptive(args.adaptive)
    for depth in (1,2):
        for seed in SEEDS:
            end=2048 if seed==11 else 512
            if not (OUT/f"fixed{depth}-{seed}-{end}.json").exists():
                subprocess.run([sys.executable,__file__,"--depth",str(depth),"--seed",str(seed)],check=True)
    for step in (512,2048):
        if not (OUT/f"reloaded-adaptive-11-{step}.json").exists():
            subprocess.run([sys.executable,__file__,"--adaptive",str(step)],check=True)


if __name__=="__main__":
    main()
