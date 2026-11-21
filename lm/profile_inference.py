"""Paired, interleaved MLX timing of deployed checkpoint inference on real contexts."""
import json
from pathlib import Path
import time
import mlx.core as mx
import numpy as np
from prepare import ROOT, RESULTS, evaluation_windows
from prototype import WideDecoder
from train import Decoder

def measure(model,scale,batch,examples):
    rng=np.random.default_rng(313+batch)
    contexts=[mx.array(examples[batch*i:batch*(i+1),:-1]) for i in range(54)]
    mx.eval(contexts)
    durations={k:[] for k in ("adaptive","1","2","4")}
    passes={k:[] for k in durations}; peaks={k:[] for k in durations}
    for idx,x in enumerate(contexts):
        for mode in rng.permutation(list(durations)):
            if idx>=6: mx.reset_peak_memory()
            then=time.perf_counter_ns()
            logits,depth=model.infer(x,str(mode))
            mx.eval(logits)
            ms=(time.perf_counter_ns()-then)/1e6
            if idx>=6:
                durations[mode].append(ms)
                passes[mode].extend(depth.tolist())
                peaks[mode].append(mx.get_peak_memory())
            del logits
    return [dict(model=scale,batch=batch,mode=k,average_passes=float(np.mean(passes[k])),
                 median_ms_per_batch=float(np.median(v)),p90_ms_per_batch=float(np.percentile(v,90)),
                 median_ms_per_example=float(np.median(v)/batch),max_peak_mlx_bytes=max(peaks[k]),
                 timings_ms=v) for k,v in durations.items()]

def main():
    all_rows=[]
    xs=evaluation_windows("test")
    for scale,constructor,weights in [("118K",Decoder,ROOT/"results/lm_followup/capstone.safetensors"),
            ("30M",WideDecoder,RESULTS/"prototype/adaptive-11-2048.safetensors")]:
        model=constructor(True); model.load_weights(str(weights)); model.eval()
        mx.eval(model.parameters())
        for size in (1,16): all_rows.extend(measure(model,scale,size,xs))
        del model
        mx.clear_cache()
    (RESULTS/"inference_profile.json").write_text(json.dumps(all_rows,indent=2)+"\n")
if __name__=="__main__": main()
