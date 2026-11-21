"""Secondary fixed-topology competitor with full visibility and nearly matched compute."""
import csv
import hashlib
from time import perf_counter

from analyze import interval
from run import BATCH,EYE,H,OUT,ROOT,SEEDS,STEPS,actions,candidates,data,decoded_actions,dispatch,fit,metrics,mx,np,write_csv

MACS = 38*24+24*23+23*4
PRIMARY_MACS = 784_465_920
BUDGET_STEPS = round(PRIMARY_MACS/(3*BATCH*MACS))


def encode(h,t,q):
    return np.concatenate([h,EYE[t].reshape(-1,32),np.eye(2,dtype=np.float32)[q]],axis=1)


def scores(p,x):
    a=mx.tanh(x@p["w1"]+p["b1"])
    a=mx.tanh(a@p["w2"]+p["b2"])
    return a@p["out"]+p["ob"]


def loss(p,x,y):
    z=scores(p,x)
    return (mx.logsumexp(z,axis=-1)-(z*y).sum(-1)).mean()


vg=mx.value_and_grad(loss)


@mx.compile
def step(p,m,v,b1,b2,x,y):
    _,grad=vg(p,x,y)
    m={k:.9*m[k]+.1*g for k,g in grad.items()}
    v={k:.999*v[k]+.001*g*g for k,g in grad.items()}
    p={k:w-.012*(m[k]/b1)/(mx.sqrt(v[k]/b2)+1e-8) for k,w in p.items()}
    return p,m,v


def dense(seed,train,testx,testy):
    h,t,q,y,_=train
    x=mx.array(encode(h,t,q))
    yy=mx.array(EYE[y])
    rng=np.random.default_rng(seed)
    def weight(i,o):
        return mx.array(rng.normal(size=(i,o)).astype(np.float32)/np.sqrt(np.float32(i)))
    p={"w1":weight(38,24),"b1":mx.zeros(24),"w2":weight(24,23),"b2":mx.zeros(23),
       "out":weight(23,4),"ob":mx.zeros(4)}
    count=sum(v.size for v in p.values())
    assert count==1607
    m={k:mx.zeros_like(w) for k,w in p.items()}
    v=dict(m)
    rows=[]
    start=perf_counter()
    for i in range(BUDGET_STEPS):
        j=i%STEPS
        if j==0:
            idx=np.arange(len(y)) if i==0 else np.random.default_rng(seed+100*i).permutation(len(y))
            epoch_x,epoch_y=x[mx.array(idx)],yy[mx.array(idx)]
            mx.eval(epoch_x,epoch_y)
        sl=slice(BATCH*j,BATCH*(j+1))
        p,m,v=step(p,m,v,mx.array(1-.9**(i+1)),mx.array(1-.999**(i+1)),epoch_x[sl],epoch_y[sl])
        mx.eval(p,m,v)
        if i+1 in (STEPS,BUDGET_STEPS):
            elapsed=perf_counter()-start
            acc,ce=metrics(np.array(scores(p,testx)),testy)
            rows.append({"seed":seed,"cell":("MLP_iso_steps" if i+1==STEPS else "MLP_iso_train_MACs"),
                "accuracy":acc,"cross_entropy":ce,"steps":i+1,"unique_train_examples":len(y),
                "examples_seen":(i+1)*BATCH,"train_linear_macs":(i+1)*BATCH*MACS*3,
                "inference_linear_macs":MACS,"parameters":count,"wall_s":elapsed})
    return rows


def main():
    pin=(OUT/"static-protocol.sha256").read_text().split()[0]
    assert hashlib.sha256((ROOT/"joint/static_control_protocol.md").read_bytes()).hexdigest()==pin
    hd,tok,q,y,_=data(995133,8192)
    rows=[]
    for seed in SEEDS:
        train=data(10000+seed,STEPS*BATCH)
        rows+=dense(seed,train,mx.array(encode(hd,tok,q)),y)
        h,t,qq,yy,_=train
        shared=[mx.array(a) for a in [h,candidates(h,t,qq),qq.astype(np.float32),EYE[yy],decoded_actions(h).astype(np.float32)]]
        for cid in ("GM","DGRM"):
            p,wall,_=fit(seed,cid,shared)
            a=actions(p,cid,hd,q)
            acc,ce=metrics(dispatch(p,hd,tok,q,a),y)
            rows.append({"seed":seed,"cell":cid,"accuracy":acc,"cross_entropy":ce,"steps":STEPS,
                "unique_train_examples":len(yy),"examples_seen":STEPS*BATCH,"train_linear_macs":PRIMARY_MACS,
                "inference_linear_macs":1536 if cid=="GM" else 1568,"parameters":1630 if cid=="GM" else 1664,
                "wall_s":wall})
    write_csv(OUT/"static_control.csv",rows)
    by={(r["seed"],r["cell"]):r for r in rows}
    diffs=[]
    for cid in ("GM","DGRM"):
        for fixed in ("MLP_iso_steps","MLP_iso_train_MACs"):
            contrasts=[100*(by[s,cid]["accuracy"]-by[s,fixed]["accuracy"]) for s in SEEDS]
            diffs.append({"contrast":cid+"_minus_"+fixed,**interval(contrasts)})
    write_csv(OUT/"static_control_deltas_pp.csv",diffs)


if __name__=="__main__":
    main()
