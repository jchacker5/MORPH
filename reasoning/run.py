"""Train and score one independent reasoning run. Use suite.py for the full matrix."""
import argparse
import gzip
import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from scipy.stats import spearmanr

from common import Decoder, ID_LENGTHS, OUT, forward_flops, generate, loss


def train(model, seed, steps, scale, extended=False, optimized=False, qualify=False):
    data_rng = np.random.default_rng(104013 + seed + (713127 if qualify else 908100 if extended else 408282 if optimized else 0))
    policy_rng = np.random.default_rng(5900 + seed)
    rate = 0.0003 if scale else 0.002
    if extended or optimized:
        peak = 0.00045 if optimized else 0.0003 * 896 / model.width
        warmup = 100 if optimized else 16
        rate = optim.join_schedules([
            optim.linear_schedule(peak * 0.1, peak, warmup),
            optim.cosine_decay(peak, steps - warmup - 1, end=peak * 0.1),
        ], [warmup])
    optimizer = optim.Adam(learning_rate=rate)
    optimizer.init(model.trainable_parameters())
    f_grad = nn.value_and_grad(model, loss)
    def get_step(passes):
        def step(x, y):
            val, gradients = f_grad(model, x, y, passes)
            if extended or optimized:
                gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            optimizer.update(model, gradients)
            return val
        state = [model.state, optimizer.state]
        return mx.compile(step, inputs=state, outputs=state)
    dispatch = {}
    batch = 24 if optimized else 8 if extended else (2 if scale else 16)
    time_cost, compute, tokens, history, cumulative = [], 0, 0, [], []
    mx.eval(model.parameters(), optimizer.state)
    for i in range(steps):
        length = 32 if scale else int(data_rng.choice(ID_LENGTHS))
        x, y, _ = generate(data_rng, batch, length)
        if model.mode == "random":
            passes = int(policy_rng.integers(1, 5))
        elif model.mode.startswith("fixed"):
            passes = int(model.mode[-1])
        else:
            passes = 4
        if passes not in dispatch:
            dispatch[passes] = get_step(passes)
        start = time.perf_counter()
        val = dispatch[passes](mx.array(x), mx.array(y))
        mx.eval(val, model.parameters(), optimizer.state)
        time_cost.append(time.perf_counter() - start)
        history.append(float(val))
        tokens += batch * length
        compute += int(3 * batch * forward_flops(model.width, length, passes, model.mode == "adaptive", train=True))
        cumulative.append(compute)
    return {"updates": steps, "train_examples": steps * batch, "train_tokens": tokens,
            "train_flops_est": compute, "step_cumulative_flops": cumulative,
            "train_step_seconds": time_cost, "train_objective": history,
            "train_peak_mlx_bytes": mx.get_peak_memory()}


def condition(seed, length, n, depths=(1, 3)):
    return generate(np.random.default_rng(seed), n, length, depths)


def cached(model, x, chunk):
    logits, hazards = [], []
    for i in range(0, len(x), chunk):
        output, hazard = model(mx.array(x[i:i+chunk]), 4, all_exits=True)
        logits.append(np.asarray(output))
        if hazard is not None:
            hazards.append(np.asarray(hazard))
    return np.concatenate(logits), np.concatenate(hazards) if hazards else None


def hard_depth(hazard, threshold):
    exits = np.column_stack([hazard >= threshold, np.ones(len(hazard), dtype=bool)])
    return exits.argmax(axis=1) + 1


def calibrate(model, scale, extended=False, optimized=False, qualify=False):
    dev = []
    for length in (32,) if scale else ID_LENGTHS:
        x, _, _ = condition(82406 + length + (569155 if qualify else 379900 if extended else 205089 if optimized else 0), length, 128 if extended else (64 if scale else 96))
        _, hazards = cached(model, x, 4 if scale else 16)
        dev.append(hazards)
    hazards = np.concatenate(dev)
    threshold = min(np.linspace(0, 1, 1001), key=lambda t: abs(hard_depth(hazards, t).mean() - 2.5))
    return float(threshold), float(hard_depth(hazards, threshold).mean())


def score(model, cond, x, y, counts, threshold, scale, record, code):
    n, length = x.shape
    chunk = 4 if scale else min(16, max(2, 4096 // length))
    adaptive = model.mode == "adaptive"
    rng = np.random.default_rng(3029 + code)
    allocation = (rng.integers(1, 5, size=n) if model.mode == "random" else
                  np.full(n, int(model.mode[-1]) if not adaptive else 4))
    predictions, chosen = [], []
    mx.reset_peak_memory()
    start = time.perf_counter()
    for i in range(0, n, chunk):
        xx = x[i:i+chunk]
        if adaptive or model.mode == "random":
            pred, depths = model.compact(xx, threshold, None if adaptive else allocation[i:i+chunk])
        else:
            logits, _ = model(mx.array(xx), int(allocation[0]))
            pred = np.asarray(logits[:, -1].argmax(axis=-1))
            depths = allocation[i:i+chunk]
        predictions.append(pred)
        chosen.append(depths)
    elapsed = time.perf_counter() - start
    physical_peak = mx.get_peak_memory()
    pred, depths = np.concatenate(predictions), np.concatenate(chosen)
    rows = []
    f = forward_flops(model.width, length, depths, adaptive)
    stats = {"name":cond, "n":n, "length":length, "correct":int((pred == y).sum()),
             "accuracy":float((pred == y).mean()), "mean_flops_est":float(f.mean()),
             "mean_depth":float(depths.mean()), "depth_hist":np.bincount(depths, minlength=5)[1:].tolist(),
             "inference_seconds":elapsed, "physical_inference_peak_mlx_bytes":physical_peak, "inference_batch":chunk}
    if adaptive:
        logits, hazards = cached(model, x, chunk)
        forced = logits.argmax(-1)
        reference_depth = hard_depth(hazards, threshold)
        perm_correct = []
        for _ in range(32):
            perm = rng.permutation(depths)
            perm_correct.append(int((forced[np.arange(n), perm-1] == y).sum()))
        raw = hard_depth(hazards, 0.5)
        raw_pred = forced[np.arange(n), raw - 1]
        stats.update({"cache_depth_disagreement":int((depths != reference_depth).sum()),
                      "cache_prediction_disagreement":int((pred != forced[np.arange(n), depths-1]).sum()),
                      "force_depth_correct":(forced == y[:,None]).sum(0).tolist(),
                      "permuted_depth_correct":perm_correct, "raw_mean_depth":float(raw.mean()),
                      "raw_correct":int((raw_pred == y).sum()),
                      "depth_count_rho":float(spearmanr(depths, counts).statistic) if len(set(depths)) > 1 else None})
    for i in range(n):
        rows.append([cond, i, length, int(counts[i]), int(y[i]), int(pred[i]), int(depths[i]),
                     int(f[i])] + (forced[i].tolist() + hazards[i].tolist() if adaptive else []))
    record.extend(rows)
    return stats


def execute(mode, seed, width, steps, scale, profile, extended=False, optimized=False, qualify=False):
    mx.random.seed(seed)
    model = Decoder(width, mode)
    if scale:
        model.set_dtype(mx.bfloat16)
    train_info = train(model, seed, steps, scale, extended, optimized, qualify)
    if profile:
        return train_info
    threshold, dev_depth = calibrate(model, scale, extended, optimized, qualify) if mode == "adaptive" else (None, None)
    conditions = [(f"id{l}", l, 384 if qualify else 256 if extended else (64 if scale else 192), (1, 3)) for l in ((32,) if scale else ID_LENGTHS)]
    if not scale:
        conditions += [(f"length{l}", l, 192 if qualify else 96, (1, 3)) for l in (256, 512, 1024)]
        conditions += [("composition64", 64, 192 if qualify else 96, (4, 6))]
    record, metrics = [], []
    for i, (name, length, n, depth_range) in enumerate(conditions):
        x,y,c = condition(663000 + i + (817239 if qualify else 908173 if extended else 291105 if optimized else 0), length, n, depth_range)
        metrics.append(score(model, name, x,y,c, threshold, scale, record,i))
    prefix = f"{'usable' if qualify else 'opt' if optimized else 'long' if extended else ('scale' if scale else 'main')}-w{width}-{mode}-s{seed}"
    if not scale and seed == 17 and (mode == "adaptive" or qualify):
        tag = f"qualified-{mode}-" if qualify else "optimized-" if optimized else ""
        model.save_weights(str(OUT / f"{tag}reasoning-seed17.safetensors"))
        (OUT / f"{tag}checkpoint.json").write_text(json.dumps({"width":width,"mode":mode,"threshold":threshold,"seed":seed},indent=2))
    columns = ["split","example","length","instructions","label","prediction","passes","matmul_flops"]
    if mode == "adaptive":
        columns += [f"forced_prediction_{d}" for d in range(1,5)] + [f"halt_probability_{d}" for d in range(1,4)]
    import csv
    with gzip.open(OUT / (prefix + ".csv.gz"), "wt") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        writer.writerows(record)
    output = {"mode":mode,"width":width,"seed":seed,"scale":scale,"extended":extended,"optimized":optimized,"qualify":qualify,"parameters":model.parameter_count(),
              "dtype":"bfloat16" if scale else "float32","threshold":threshold,"dev_depth":dev_depth,
              **train_info,"splits":metrics}
    if mode == "adaptive" and (extended or optimized) and not qualify:
        from sensitivity import write
        write(model, seed, extended, optimized, record, threshold)
    return output


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["fixed1","fixed2","fixed3","fixed4","untied4","random","adaptive"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--steps", type=int,required=True)
    parser.add_argument("--scale",action="store_true")
    parser.add_argument("--extended",action="store_true")
    parser.add_argument("--optimized",action="store_true")
    parser.add_argument("--qualify",action="store_true")
    parser.add_argument("--profile",action="store_true")
    args=parser.parse_args()
    if args.steps < 1 or args.width < 1 or args.width % min(args.width,64) != 0:
        parser.error("Supply positive steps and a width divisible by the attention head size.")
    if args.extended and not args.scale:
        parser.error("The longer scaled study requires --scale --extended.")
    if args.optimized and args.scale:
        parser.error("The longer small-model study uses --optimized without --scale.")
    if args.qualify and (not args.optimized or args.mode not in {"adaptive","fixed2"}):
        parser.error("Qualification requires --optimized and either adaptive or fixed2 mode.")
    result=execute(**vars(args))
    prefix = ("profile" if args.profile else "usable" if args.qualify else "opt" if args.optimized else "long" if args.extended else "scale" if args.scale else "main")
    file = OUT / f"{prefix}-w{args.width}-{args.mode}-s{args.seed}.json"
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(file)
    print(file, flush=True)
