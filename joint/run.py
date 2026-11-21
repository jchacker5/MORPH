"""Frozen 2^4 controller factorial. Run from project root using .venv/bin/python joint/run.py."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "joint"
SEEDS = [41, 42, 43]
BATCH, STEPS, TEST, H, K = 64, 240, 4096, 24, 16
AXES = "DGRM"
HA = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], np.float32) / 2
PAIR = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]]])
PERMS = np.array([[1,2,3,0],[0,2,3,1]])
EYE = np.eye(4, dtype=np.float32)
mx.set_default_device(mx.cpu)


def data(seed, count):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(count, 4))
    header = ((2*bits-1) @ HA + rng.normal(0, .15, (count, 4))).astype(np.float32)
    tokens = rng.integers(0, 4, (count, 8))
    query = rng.integers(0, 2, count)
    d, g, r, m = bits.T
    addresses = PAIR[g, query] + 4*m[:, None]
    start = np.take_along_axis(tokens, addresses, axis=1).sum(1) % 4
    one = PERMS[r, start]
    labels = np.where(d, PERMS[r, PERMS[r, one]], one)
    return header, tokens, query, labels, bits


def candidates(head, tok, q):
    """Training-only enumeration of paths, in memory-major, grouping-minor order."""
    stacks = []
    rows = np.arange(len(q))[:, None]
    for bank, grouping in itertools.product(range(2), repeat=2):
        ids = tok[rows, bank*4 + PAIR[grouping, q]]
        stacks.append(np.concatenate([EYE[ids].sum(1), head], axis=-1))
    return np.stack(stacks, axis=1)


def init(seed):
    rng = np.random.default_rng(seed)
    def w(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)
    return {"c1":w((4,K), .35), "c1b":np.zeros(K,np.float32),
            "c2":w((K,4), .2), "c2b":np.zeros(4,np.float32),
            "stem":w((8,H),1/np.sqrt(8)), "sb":np.zeros(H,np.float32),
            "expert":w((2,H,H),1/np.sqrt(H)), "eb":np.zeros((2,1,H),np.float32),
            "out":w((H,4),1/np.sqrt(H)), "ob":np.zeros(4,np.float32)}


def decoded_actions(head):
    scores = head @ HA.T
    bits = scores > 0
    bits[:,0] = quota(scores[:,0])
    return bits.astype(np.int32)


def quota(scores):
    blocks = scores.reshape(-1, BATCH)
    order = np.argsort(blocks, axis=1, kind="stable")
    out = np.zeros_like(blocks, dtype=np.int32)
    np.put_along_axis(out, order[:, BATCH//2:], 1, axis=1)
    return out.ravel()


def train_loss(p, hd, cand, query, labels, use, hand, informed):
    score = mx.tanh(hd @ p['c1'] + p['c1b']) @ p['c2'] + p['c2b']
    probs = mx.sigmoid(score)
    # For fixed D the probability of middle depth is one.
    fixed = mx.stack([mx.zeros_like(query), mx.zeros_like(query), query, mx.ones_like(query)], axis=1)
    action = hand*informed + (1-informed)*(use*probs + (1-use)*fixed)
    d,g,r,m = (action[:,j] for j in range(4))
    enabled_d = informed + (1-informed)*use[0]
    ds = mx.stack([(1-d)*enabled_d, mx.ones_like(d)*(1-enabled_d), d*enabled_d], axis=0)
    gs = mx.stack([(1-g)*(1-m),g*(1-m),(1-g)*m,g*m], axis=-1)
    rs = mx.stack([1-r,r], axis=0)
    base = mx.tanh(cand @ p['stem'] + p['sb']).reshape(1,-1,H)
    z = mx.broadcast_to(base,(2,base.shape[1],H))
    losses = []
    for _ in range(3):
        z = mx.tanh(z @ p['expert'] + p['eb'] + base)
        logits = (z @ p['out'] + p['ob']).reshape(2,-1,4,4)
        # labels are one-hot.
        nll = mx.logsumexp(logits, axis=-1) - (logits * labels[None,:,None,:]).sum(-1)
        losses.append((nll * gs[None,:,:] * rs[:,:,None]).sum(axis=(0,2)))
    risk = (mx.stack(losses)*ds).sum(0).mean()
    regularizer = .25*(probs[:,0].mean()-.5)**2 * use[0]*(1-informed)
    return risk + regularizer


vg = mx.value_and_grad(train_loss)


@mx.compile
def update(p, first, second, bc1, bc2, *batch):
    loss, grad = vg(p, *batch)
    first = {k:.9*first[k]+.1*g for k,g in grad.items()}
    second = {k:.999*second[k]+.001*g*g for k,g in grad.items()}
    p = {k:v-.012*(first[k]/bc1)/(mx.sqrt(second[k]/bc2)+1e-8) for k,v in p.items()}
    return p,first,second,loss


def fit(seed, run_id, shared):
    start = time.perf_counter()
    p = {k:mx.array(v) for k,v in init(seed).items()}
    m = {k:mx.zeros_like(v) for k,v in p.items()}
    v = dict(m)
    use = mx.array([float(a in run_id) for a in AXES])
    informed = mx.array(float(run_id == 'hand'))
    for i in range(STEPS):
        part = slice(i*BATCH,(i+1)*BATCH)
        args = [a[part] for a in shared]
        p,m,v,loss = update(p,m,v,mx.array(1-.9**(i+1)),mx.array(1-.999**(i+1)),
                            *args[:4],use,args[4],informed)
        mx.eval(p,m,v,loss)
    return {k:np.array(a) for k,a in p.items()},time.perf_counter()-start,float(loss.item())


def actions(p, run_id, head, q):
    score = np.tanh(head @ p['c1'] + p['c1b']) @ p['c2'] + p['c2b']
    chosen = (score > 0).astype(np.int32)
    chosen[:,0] = quota(score[:,0])
    if run_id == "hand":
        chosen = decoded_actions(head)
    else:
        fixed = np.column_stack([np.zeros_like(q), np.zeros_like(q), q, np.ones_like(q)])
        for i,a in enumerate(AXES):
            if a not in run_id:
                chosen[:,i] = fixed[:,i]
    chosen[:,0] = 1+2*chosen[:,0] if 'D' in run_id or run_id=='hand' else 2
    return chosen


def dispatch(p, head, tok, q, chosen):
    """Single-bank, two-summary physical dispatch. No unselected expert or pass is run."""
    n = len(q)
    d,g,r,m = chosen.T
    retained = np.take_along_axis(tok,4*m[:,None]+np.arange(4),axis=1)
    paired = retained[np.arange(n)[:,None,None], PAIR[g]]
    pooled = EYE[paired].sum(2)
    chosen_pair = pooled[np.arange(n), q]
    base = np.tanh(np.concatenate([chosen_pair,head],axis=1) @ p['stem'] + p['sb'])
    z = base.copy()
    for expert in (0,1):
        for level in range(1,4):
            rows = np.flatnonzero((r==expert)&(d>=level))
            if len(rows):
                z[rows] = np.tanh(z[rows] @ p['expert'][expert]+p['eb'][expert]+base[rows])
    return z @ p['out'] + p['ob']


def ref(p, head, tok, q, chosen):
    cand = candidates(head,tok,q)
    d,g,r,m = chosen.T
    n = len(q)
    base = np.tanh(cand @ p['stem'] + p['sb'])
    scores = np.empty((n,4),np.float32)
    for expert in (0,1):
        z = base
        for level in range(1,4):
            z = np.tanh(z @ p['expert'][expert]+p['eb'][expert]+base)
            rows = np.flatnonzero((r==expert)&(d==level))
            scores[rows] = z[rows,2*m[rows]+g[rows]] @ p['out'] + p['ob']
    return scores


def metrics(logits,y):
    z = logits-logits.max(-1,keepdims=True)
    ce = np.log(np.exp(z).sum(-1))-z[np.arange(len(y)),y]
    return float(np.mean(logits.argmax(-1)==y)),float(ce.mean())


def summarize_run(p, seed, cid, test, wall, last):
    hd,tok,q,y,bits = test
    pred = actions(p,cid,hd,q)
    logits = dispatch(p,hd,tok,q,pred)
    np.testing.assert_allclose(logits[:128],ref(p,hd[:128],tok[:128],q[:128],pred[:128]),atol=3e-5,rtol=1e-5)
    assert pred[:,0].mean() == 2.
    acc,ce = metrics(logits,y)
    heads = 0 if cid=="hand" else sum(a in cid for a in AXES)
    count = sum(v.size for v in p.values())
    ctrl = (4+1)*K+(K+1)*4
    base_macs = 8*H+2*H*H+H*4
    norm_macs = base_macs+4*K+K*4
    # Header decoding for all four hand policy scores is a 4x4 product.
    compact_macs = base_macs + (16 if cid=="hand" else (4*K+K*heads if heads else 0))
    forward_train = 4*8*H+3*4*2*H*H+3*4*2*H*4+4*K+4*K
    run = {"seed":seed,"cell":cid,"accuracy":acc,"cross_entropy":ce,"last_train_objective":last,
        "steps":STEPS,"train_examples":STEPS*BATCH,"test_examples":TEST,"train_seconds":wall,
        "mean_passes":float(pred[:,0].mean()),"allocated_parameters":count,
        "effective_trainable_parameters":count-ctrl+((4+1)*K+(K+1)*heads if heads else 0),
        "standardized_inference_linear_macs":norm_macs if cid!="hand" else norm_macs+16,
        "compact_inference_linear_macs":compact_macs,
        "nominal_dense_train_linear_macs":3*forward_train*STEPS*BATCH}
    latency = []
    for _ in range(5):
        start = time.perf_counter()
        for j in range(0,TEST,BATCH):
            sl = slice(j,j+BATCH)
            allocated = actions(p,cid,hd[sl],q[sl])
            dispatch(p,hd[sl],tok[sl],q[sl],allocated)
        latency.append((time.perf_counter()-start)/TEST*1e6)
    run['median_us_per_example_at_batch64'] = float(np.median(latency))
    extra = []
    if cid=="DGRM":
        for i,a in enumerate(AXES):
            activated = (pred[:,i]==3 if i==0 else pred[:,i]).astype(int)
            agree = np.mean(activated==bits[:,i])
            run[a+"_activation_rate"] = float(activated.mean())
            run[a+"_teacher_agreement"] = float(agree)
            if a=="R":
                run["R_permutation_invariant_agreement"] = float(max(agree,1-agree))
        rng = np.random.default_rng(8675309)
        perm = np.concatenate([j+rng.permutation(BATCH) for j in range(0,TEST,BATCH)])
        for label,cols in [('tuple',range(4)),*[(a,[i]) for i,a in enumerate(AXES)]]:
            changed = pred.copy()
            changed[:,cols] = pred[perm][:,cols]
            ai,ci = metrics(dispatch(p,hd,tok,q,changed),y)
            extra.append({"seed":seed,"control":label,"accuracy":ai,"cross_entropy":ci,
                          "natural_minus_intervention":acc-ai,"test_examples":TEST})
        for depth in (1,3):
            changed = pred.copy()
            changed[:,0] = depth
            probe = dispatch(p,hd,tok,q,changed)
            for teacher_d in (0,1):
                rows = bits[:,0]==teacher_d
                ai,ci=metrics(probe[rows],y[rows])
                extra.append({"seed":seed,"control":f"force{depth}_teacher{1+2*teacher_d}",
                              "accuracy":ai,"cross_entropy":ci,"test_examples":int(rows.sum())})
    return run,extra


def write_csv(path,rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    frozen = (OUT/'protocol.sha256').read_text().split()[0]
    assert hashlib.sha256((ROOT/'docs/joint_protocol.md').read_bytes()).hexdigest()==frozen
    test = data(910274,TEST)
    if args.verify_only:
        p = init(5)
        for cid in ('fixed','DGRM','hand'):
            summarize_run(p,5,cid,test,0,0)
        print("Matched two-pass cohort totals and verified real dispatch against dense reference.")
        return
    runs,interventions = [],[]
    cells = [''.join(a for a,b in zip(AXES,mask) if b) or 'fixed'
             for mask in itertools.product([0,1],repeat=4)] + ['hand']
    from importlib.metadata import version
    (OUT/'environment.json').write_text(json.dumps({"protocol_sha256":frozen,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy":np.__version__,"mlx":version("mlx"),"device":str(mx.default_device()),
        "os":platform.platform(),"training_seeds":SEEDS,"test_seed":910274},indent=2))
    for seed in SEEDS:
        hd,tok,q,y,_ = data(seed+10000,STEPS*BATCH)
        shared = [mx.array(a) for a in [hd,candidates(hd,tok,q),q.astype(np.float32),
                  EYE[y],decoded_actions(hd).astype(np.float32)]]
        for cid in cells:
            p,wall,last = fit(seed,cid,shared)
            row, probes = summarize_run(p,seed,cid,test,wall,last)
            runs.append(row)
            interventions.extend(probes)
            write_csv(OUT/'fits.csv',runs)
            if interventions:
                write_csv(OUT/'interventions.csv',interventions)
            if cid=="DGRM" and seed==SEEDS[0]:
                np.savez(OUT/'joint_seed41.npz', **p)
            print(f"seed {seed} cell {cid} complete in {wall:.2f}s",flush=True)


if __name__=="__main__":
    main()
