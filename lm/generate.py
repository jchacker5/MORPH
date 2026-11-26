"""Generate bytes from a saved recurrent MORPH language model."""
import argparse
import math
from numbers import Integral, Real
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results/lm"


def validate_inputs(prompt, length, temperature):
    """Reject invalid requests before importing MLX or opening a checkpoint."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Provide a nonempty text prompt")
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("temperature must be finite and nonnegative")

def generate(weights,prompt,length,temperature,seed,wide=False):
    validate_inputs(prompt, length, temperature)
    data=list(prompt.encode("utf-8"))
    if length == 0:
        return bytes(data)
    import mlx.core as mx
    from prepare import WINDOW
    from train import Decoder
    if wide:
        from prototype import WideDecoder
        model=WideDecoder(True)
    else:
        model=Decoder(True)
    model.load_weights(str(weights))
    model.eval()
    mx.random.seed(seed)
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
    try:
        validate_inputs(a.prompt, a.bytes, a.temperature)
    except ValueError as error:
        p.error(str(error))
    print(generate(a.weights or default,a.prompt,a.bytes,a.temperature,a.seed,not a.small).decode("utf-8",errors="replace"))
