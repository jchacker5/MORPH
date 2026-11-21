"""MLX adaptive prefix-conditioned causal decoder, search trial and final-training entry points."""
from functools import partial
import argparse
import json
import math
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from prepare import (BATCH, HEADS, PREFIX, RESULTS, VOCAB, WIDTH, WINDOW,
                     evaluation_windows, split_data, windows_at)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.norm = nn.LayerNorm(WIDTH)
        self.expand = nn.Linear(WIDTH, 3 * WIDTH)
        self.contract = nn.Linear(3 * WIDTH, WIDTH)

    def __call__(self, h):
        n, t, _ = h.shape
        def heads(x):
            return x.reshape(n, t, HEADS, WIDTH // HEADS).transpose(0, 2, 1, 3)
        q, k, v = [heads(x) for x in mx.split(self.qkv(self.pre(h)), 3, axis=-1)]
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(WIDTH // HEADS)**-0.5, mask="causal")
        h = h + self.proj(a.transpose(0, 2, 1, 3).reshape(n, t, WIDTH))
        return h + self.contract(nn.gelu(self.expand(self.norm(h))))

class Decoder(nn.Module):
    def __init__(self, adaptive, bias=0.0):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, WIDTH)
        self.block = Block()
        self.out = nn.LayerNorm(WIDTH)
        if adaptive:
            self.halt = nn.Linear(WIDTH, 1)
            self.halt.bias = mx.array([float(bias)])

    def embed(self, x):
        phase = mx.arange(x.shape[1])[:, None] * mx.exp(-math.log(10000) * mx.arange(WIDTH // 2) / (WIDTH // 2))
        return self.emb(x) * math.sqrt(WIDTH) + mx.concatenate([mx.sin(phase), mx.cos(phase)], -1)

    def logits(self, h):
        return self.emb.as_linear(self.out(h))

    def hazard(self, h):
        return mx.sigmoid(self.halt(self.out(h[:, :PREFIX]).mean(axis=1))[:, 0])

    def objective(self, data, mode, depth, cost):
        x, y = data[:, :-1], data[:, PREFIX+1:]
        h = self.embed(x)
        survival, loss = 1.0, 0.0
        for step in range(1, depth+1):
            h = self.block(h)
            if mode == "adaptive":
                p = self.hazard(h) if step < depth else 1.0
                weight = survival * p
                survival = survival * (1-p)
            elif step == depth:
                weight = 1.0
            else:
                continue
            logits = self.logits(h[:, PREFIX:])
            ce = nn.losses.cross_entropy(logits, y).mean(axis=1)
            loss = loss + weight * (ce + (cost * step if mode == "adaptive" else 0))
        return loss.mean()

    def infer(self, x, mode, threshold=.5, rng=None, last=False):
        # Compact the batch after each stop, so exited examples skip subsequent blocks.
        n = x.shape[0]
        ids = np.arange(n)
        if mode == "random":
            limits = rng.integers(1, 5, size=n)
        elif mode != "adaptive":
            limits = np.full(n, int(mode))
        h = self.embed(x)
        depths = np.empty(n, dtype=np.int32)
        chunks, indices = [], []
        for step in range(1, 5):
            h = self.block(h)
            if mode == "adaptive":
                stop = np.asarray(self.hazard(h)) >= threshold if step < 4 else np.ones(len(ids), bool)
            else:
                stop = limits[ids] == step
            if stop.any():
                exited = h if stop.all() else h[mx.array(np.flatnonzero(stop))]
                out = self.logits(exited[:, -1:] if last else exited[:, PREFIX:])
                chunks.append(out)
                indices.extend(ids[stop])
                depths[ids[stop]] = step
            if stop.all():
                break
            if stop.any():
                h = h[mx.array(np.flatnonzero(~stop))]
                ids = ids[~stop]
        logits = mx.concatenate(chunks) if len(chunks)>1 else chunks[0]
        if indices != list(range(n)):
            logits = logits[mx.array(np.argsort(indices))]
        return logits, depths

def flops(depth, adaptive, training=False):
    # Dense-equivalent attention, counting the full square even with a causal mask.
    block = 2 * WINDOW * (10 * WIDTH**2) + 4 * WINDOW**2 * WIDTH
    head = 2 * (WINDOW-PREFIX) * WIDTH * VOCAB
    gates = (3 if training else np.minimum(depth, 3)) * 2 * WIDTH if adaptive else 0
    return (block * depth + head * (4 if adaptive and training else 1) + gates) * (3 if training else 1)

def score(model, mode, examples, threshold=.5):
    rng = np.random.default_rng(701)
    total, hist = 0.0, np.zeros(4, dtype=int)
    for i in range(0, len(examples), 32):
        rows = mx.array(examples[i:i+32])
        logits, depths = model.infer(rows[:,:-1],mode,threshold,rng)
        total += nn.losses.cross_entropy(logits, rows[:,PREFIX+1:]).sum().item()
        hist += np.bincount(depths, minlength=5)[1:]
    d = np.arange(1,5)
    return dict(bpb=total/(len(examples)*(WINDOW-PREFIX)*math.log(2)), histogram=hist.tolist(),
                examples=len(examples), scored_bytes=len(examples)*(WINDOW-PREFIX), mean_depth=float(hist @ d / hist.sum()),
                inference_flops_per_window=float(hist @ flops(d,mode=="adaptive")/hist.sum()))

def fit(mode, seed, lr, cost, bias, steps=None, seconds=None):
    mx.random.seed(seed)
    rng, router = np.random.default_rng(seed), np.random.default_rng(9300+seed)
    model = Decoder(mode=="adaptive", bias)
    opt = optim.AdamW(learning_rate=lr, weight_decay=.01)
    gradients = nn.value_and_grad(model, model.objective)
    def make_update(depth):
        @partial(mx.compile, inputs=[model.state,opt.state],outputs=[model.state,opt.state])
        def update(data):
            loss, grads = gradients(data,mode,depth,cost)
            opt.update(model, grads)
            return loss
        return update
    depths = range(1,5) if mode=="random" else [4 if mode=="adaptive" else int(mode)]
    funcs = {d:make_update(d) for d in depths}
    # Compile/warm every path on fake bytes without taking an optimizer step on real training data.
    # Rewind parameters and optimizer state, preserving compiled shapes.
    from mlx.utils import tree_map
    original = tree_map(lambda x: x, model.parameters())
    opt.init(model.trainable_parameters())
    original_opt = tree_map(lambda x: x, opt.state)
    for fn in funcs.values():
        loss = fn(mx.zeros((BATCH,WINDOW+1),mx.int32))
        mx.eval(loss,model.parameters(),opt.state)
    model.update(original)
    opt.state.update(original_opt)
    data = split_data("train")
    mx.reset_peak_memory()
    started, step, work = time.perf_counter(), 0, 0.0
    while (steps is not None and step<steps) or (seconds is not None and time.perf_counter()-started < seconds):
        starts = rng.integers(0,len(data)-WINDOW,size=BATCH)
        depth = int(router.integers(1,5)) if mode=="random" else depths[0]
        loss = funcs[depth](mx.array(windows_at(data,starts)))
        mx.eval(loss,model.parameters(),opt.state)
        step += 1
        work += BATCH * flops(depth,mode=="adaptive",True)
    elapsed = time.perf_counter()-started
    return model, dict(mode=mode,seed=seed,lr=lr,cost=cost,bias=bias,steps=step,seconds=elapsed,
                       input_bytes=step*BATCH*WINDOW,scored_train_bytes=step*BATCH*(WINDOW-PREFIX),
                       train_flops=float(work),training_peak_mlx_bytes=mx.get_peak_memory(),last_loss=float(loss.item()),
                       parameters=sum(p.size for _,p in tree_flatten(model.parameters())))

def run(a):
    RESULTS.mkdir(parents=True,exist_ok=True)
    model, row = fit(a.mode,a.seed,a.lr,a.cost,a.bias,a.steps,a.seconds)
    which = "val" if a.seconds is not None else "test"
    examples = evaluation_windows(which)
    row[which] = score(model,a.mode,examples)
    if which=="test":
        if a.mode=="adaptive":
            row["threshold_sensitivity"] = {str(t):score(model,a.mode,examples,t) for t in [.35,.65]}
        rng = np.random.default_rng(701)
        sample = mx.array(examples[:1,:-1])
        for _ in range(4): mx.eval(model.infer(sample,a.mode,rng=rng)[0])
        times=[]
        for i in range(40):
            x = mx.array(examples[i:i+1,:-1])
            tick=time.perf_counter()
            mx.eval(model.infer(x,a.mode,rng=rng)[0])
            times.append((time.perf_counter()-tick)*1000)
        row["batch1_median_ms"] = float(np.median(times))
    if a.weights:
        model.save_weights(str(a.weights))
    Path(a.output).write_text(json.dumps(row,indent=2)+"\n")
    print(json.dumps(row,sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["1","2","4","random","adaptive"],default="adaptive")
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--lr",type=float,default=.002)
    ap.add_argument("--cost",type=float,default=.02)
    ap.add_argument("--bias",type=float,default=0)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--steps",type=int)
    g.add_argument("--seconds",type=float)
    ap.add_argument("--weights")
    ap.add_argument("--output",required=True)
    run(ap.parse_args())
