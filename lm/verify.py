"""Check held-out scores after reloading, true compaction and prefix causality."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from prepare import PREFIX, RESULTS, WINDOW, evaluation_windows
from train import Decoder, score
from generate import generate
from prototype import WideDecoder

def checks(model, examples):
    x=mx.array(examples[:12,:-1])
    y=np.asarray(x).copy()
    y[:,35:]=(y[:,35:]+73)%256
    logits,depth=model.infer(x,"adaptive")
    changed,changed_depth=model.infer(mx.array(y),"adaptive")
    max_delta=float(mx.max(mx.abs(logits[:,:35-PREFIX]-changed[:,:35-PREFIX])).item())
    if max_delta>1e-4 or np.any(depth!=changed_depth):
        raise ValueError(f"Future-byte leakage (delta {max_delta})")
    hs=model.embed(x)
    history=[]
    for d in range(1,5):
        hs=model.block(hs)
        history.append(np.asarray(model.logits(hs[:,PREFIX:]).astype(mx.float32)))
    dense=np.stack(history,axis=1)[np.arange(len(depth)),depth-1]
    discrepancy=float(np.max(abs(np.asarray(logits)-dense)))
    return {"max_prefix_logit_change_after_suffix_mutation":max_delta,"max_selected_exit_logit_difference":discrepancy,"checked_examples":len(depth)}

def main():
    examples=evaluation_windows("test")
    reports=[]
    for wide in [False,True]:
        steps=2048 if wide else 320
        label="30M" if wide else "118K"
        path=RESULTS/("prototype/adaptive-11-2048.safetensors" if wide else "adaptive-11.safetensors")
        result=RESULTS/("prototype/adaptive-11-2048.json" if wide else "final/adaptive-11.json")
        model=WideDecoder(True) if wide else Decoder(True)
        model.load_weights(str(path))
        report=checks(model,examples)
        if report["max_selected_exit_logit_difference"]>(.16 if wide else 1e-4):
            raise ValueError("Hard exit and dense reference differ")
        evaluation=score(model,"adaptive",examples)
        saved=json.loads(result.read_text())["test"]
        if abs(evaluation["bpb"]-saved["bpb"])>1e-5 or evaluation["histogram"]!=saved["histogram"]:
            raise ValueError("Checkpoint did not reproduce final metric")
        generated=generate(path,"KING HENRY:",160,.8,73,wide)
        again=generate(path,"KING HENRY:",160,.8,73,wide)
        assert generated==again
        (RESULTS/f"sample-{label}.txt").write_text(generated.decode(errors="replace")+"\n")
        report.update(scale=label,steps=steps,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      file_bytes=path.stat().st_size,reloaded_bpb=evaluation["bpb"],reloaded_exit_histogram=evaluation["histogram"],
                      identical_seeded_sample=True,sample_sha256=hashlib.sha256(generated).hexdigest())
        reports.append(report)
    # Force all four bins within one mixed batch using an auditable deterministic
    # test double. Even a collapsed trained policy must exercise the gather path.
    class DispatchCheck:
        infer = Decoder.infer

        def embed(self,x):
            return x[:,:,None].astype(mx.float32)
        def block(self,h):
            return h+10
        def hazard(self,h):
            code=h[:,0,0]
            return (mx.floor(code/10)>=(code%10)+1).astype(mx.float32)
        def logits(self,h):
            return h
    toy=DispatchCheck()
    tags=mx.array([3,1,0,2,1,3,2,0])
    toy_x=mx.broadcast_to(tags[:,None],(8,WINDOW))
    outputs,exits=toy.infer(toy_x,"adaptive",last=True)
    expected=np.asarray(tags)+1
    assert np.array_equal(exits,expected) and np.array_equal(np.asarray(outputs).reshape(-1),np.asarray(tags)+10*expected)
    (RESULTS/"verification.json").write_text(json.dumps({"checkpoints":reports,"mixed_reordering_exits":exits.tolist()},indent=2)+"\n")

if __name__=="__main__": main()
