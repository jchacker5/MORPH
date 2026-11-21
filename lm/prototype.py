"""Thirty-million-parameter recurrent LM. FP32 master weights, BF16 matrix products."""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from prepare import RESULTS, WINDOW, PREFIX, VOCAB, evaluation_windows, split_data, windows_at
from train import Decoder, score

D, H, B = 1728,16,4
FOLDER = RESULTS/"prototype"

class BFLinear(nn.Linear):
    def __call__(self,x):
        weights = self.weight.astype(mx.bfloat16)
        out = x.astype(mx.bfloat16) @ weights.T
        return out + self.bias.astype(out.dtype) if "bias" in self else out

class WideBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre=nn.LayerNorm(D)
        self.qkv=BFLinear(D,3*D,bias=False)
        self.proj=BFLinear(D,D,bias=False)
        self.norm=nn.LayerNorm(D)
        self.expand=BFLinear(D,3*D)
        self.contract=BFLinear(3*D,D)

    def __call__(self,h):
        n,t,_=h.shape
        q,k,v=[z.reshape(n,t,H,D//H).transpose(0,2,1,3) for z in mx.split(self.qkv(self.pre(h)),3,axis=-1)]
        a=mx.fast.scaled_dot_product_attention(q,k,v,scale=(D//H)**-.5,mask="causal")
        h=h+self.proj(a.transpose(0,2,1,3).reshape(n,t,D))
        return h+self.contract(nn.gelu(self.expand(self.norm(h))))

class WideDecoder(Decoder):
    def __init__(self,adaptive,bias=0.):
        nn.Module.__init__(self)
        self.emb=nn.Embedding(VOCAB,D)
        self.block=WideBlock()
        self.out=nn.LayerNorm(D)
        if adaptive:
            self.halt=nn.Linear(D,1)
            self.halt.bias=mx.array([float(bias)])

    def embed(self,x):
        phase=mx.arange(x.shape[1])[:,None]*mx.exp(-math.log(10000)*mx.arange(D//2)/(D//2))
        return (self.emb(x)*math.sqrt(D)+mx.concatenate([mx.sin(phase),mx.cos(phase)],axis=-1)).astype(mx.bfloat16)

    def logits(self,h):
        return (self.out(h).astype(mx.bfloat16) @ self.emb.weight.astype(mx.bfloat16).T).astype(mx.float32)

def cost_for(depth,adaptive,training=False):
    blocks=20*WINDOW*D**2+4*WINDOW**2*D
    head=2*(WINDOW-PREFIX)*D*VOCAB
    gates=(3 if training else np.minimum(depth,3))*2*D if adaptive else 0
    return (blocks*depth+head*(4 if training and adaptive else 1)+gates)*(3 if training else 1)

def evaluate(model,mode,examples):
    row=score(model,mode,examples)
    hist=np.asarray(row["histogram"])
    row["inference_flops_per_window"]=float(hist@cost_for(np.arange(1,5),mode=="adaptive")/hist.sum())
    return row

def train_seed(mode,seed,cfg):
    adaptive=mode=="adaptive"
    mx.random.seed(seed)
    model=WideDecoder(adaptive,cfg["bias"])
    opt=optim.AdamW(learning_rate=.0003,weight_decay=.01)
    opt.init(model.trainable_parameters())
    gradient=nn.value_and_grad(model,model.objective)
    @partial(mx.compile,inputs=[model.state,opt.state],outputs=[model.state,opt.state])
    def update(rows):
        loss,grad=gradient(rows,mode,4,cfg["cost"])
        opt.update(model,grad)
        return loss
    mx.eval(model.parameters(),opt.state)
    data=split_data("train")
    test=evaluation_windows("test")
    rng=np.random.default_rng(seed)
    tick=time.perf_counter()
    mx.reset_peak_memory()
    elapsed=0.
    params=sum(p.size for _,p in tree_flatten(model.parameters()))
    for step in range(1,(2048 if seed==11 else 512)+1):
        starts=rng.integers(0,len(data)-WINDOW,size=B)
        loss=update(mx.array(windows_at(data,starts)))
        mx.eval(loss,model.parameters(),opt.state)
        if step%128==0: print(f"{mode=} {seed=} {step=} elapsed {time.perf_counter()-tick:.2f}s",flush=True)
        if step not in (512,2048): continue
        elapsed+=time.perf_counter()-tick
        row=dict(mode=mode,seed=seed,steps=step,lr=.0003,parameters=params,cost=cfg["cost"],bias=cfg["bias"],
                 input_bytes=step*B*WINDOW,scored_train_bytes=step*B*(WINDOW-PREFIX),
                 training_seconds_including_first_compile=elapsed,training_peak_mlx_bytes=mx.get_peak_memory(),
                 train_flops=int(step*B*cost_for(4,adaptive,True)),last_train_loss=float(loss.item()))
        if mode=="adaptive" and seed==11:
            model.save_weights(str(FOLDER/f"adaptive-11-{step}.safetensors"))
        row["test"]=evaluate(model,mode,test)
        row["forced_exits"]={str(d):evaluate(model,str(d),test) for d in [1,2,3,4]}
        durations=[]
        for i in range(44):
            x=mx.array(test[i:i+1,:-1])
            t=time.perf_counter()
            mx.eval(model.infer(x,mode)[0])
            durations.append(time.perf_counter()-t)
        row["batch1_median_ms"]=float(1000*np.median(durations[4:]))
        (FOLDER/f"{mode}-{seed}-{step}.json").write_text(json.dumps(row,indent=2)+"\n")
        print("Saved",mode,seed,step,flush=True)
        mx.reset_peak_memory()
        tick=time.perf_counter()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["4","adaptive"],required=True)
    ap.add_argument("--seed",type=int,choices=[11,23,37],required=True)
    args=ap.parse_args()
    cfg=json.loads((RESULTS/"frozen.json").read_text())["config"]
    FOLDER.mkdir(exist_ok=True)
    train_seed(args.mode,args.seed,cfg)
