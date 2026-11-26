"""Generate bytes from a saved recurrent MORPH language model."""
import argparse
import math
from pathlib import Path
import mlx.core as mx
from prepare import RESULTS, WINDOW
from train import Decoder

def generate(weights,prompt,length,temperature,seed,wide=False):
    if length < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("length and temperature must be finite and nonnegative")
    if wide:
        from prototype import WideDecoder
        model=WideDecoder(True)
    else:
        model=Decoder(True)
    model.load_weights(str(weights))
    model.eval()
    mx.random.seed(seed)
    data=list(prompt.encode("utf-8"))
    if not data: raise ValueError("Provide a nonempty prompt")
    for _ in range(length):
        logits,_=model.infer(mx.array([data[-WINDOW:]]),"adaptive",last=True)
        scores=logits[0,-1]
        token=mx.random.categorical(scores/temperature) if temperature>0 else mx.argmax(scores)
        data.append(int(token.item()))
    return bytes(data)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prompt",default="KING HENRY:")
    p.add_argument("--bytes",type=int,default=160)
    p.add_argument("--temperature",type=float,default=.8)
    p.add_argument("--seed",type=int,default=73)
    p.add_argument("--small",action="store_true")
    p.add_argument("--weights",type=Path)
    a=p.parse_args()
    default=(RESULTS/"adaptive-11.safetensors" if a.small else RESULTS/"prototype/adaptive-11-2048.safetensors")
    if a.bytes<0 or not math.isfinite(a.temperature) or a.temperature<0: p.error("bytes and temperature must be nonnegative")
    print(generate(a.weights or default,a.prompt,a.bytes,a.temperature,a.seed,not a.small).decode("utf-8",errors="replace"))
