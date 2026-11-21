"""Reconstruct GM seed 41, verify the frozen test result, then check single-example inference."""
import csv
import hashlib
import json
from pathlib import Path

from run import BATCH, EYE, OUT, ROOT, STEPS, TEST, actions, candidates, data, decoded_actions, dispatch, fit, metrics, mx, np, ref
from infer import predict


def main():
    pin=(OUT/"usage-protocol.sha256").read_text().split()[0]
    assert hashlib.sha256((ROOT/"joint/usage_check.md").read_bytes()).hexdigest()==pin
    with (OUT/"fits.csv").open() as stream:
        receipt=next(r for r in csv.DictReader(stream) if r["cell"]=="GM" and int(r["seed"])==41)
    h,t,q,y,_=data(10041,STEPS*BATCH)
    train=[mx.array(a) for a in [h,candidates(h,t,q),q.astype(np.float32),EYE[y],decoded_actions(h).astype(np.float32)]]
    p,_,_=fit(41,"GM",train)
    h,t,q,y,_=data(910274,TEST)
    a=actions(p,"GM",h,q)
    z=dispatch(p,h,t,q,a)
    np.testing.assert_allclose(z[:128],ref(p,h[:128],t[:128],q[:128],a[:128]),atol=3e-5,rtol=1e-5)
    acc,ce=metrics(z,y)
    assert acc==float(receipt["accuracy"]) and abs(ce-float(receipt["cross_entropy"]))<1e-6
    dest=OUT/"gm_seed41.npz"
    np.savez(dest,**p)
    with np.load(dest) as saved:
        weights=dict(saved)
    h,t,q,y,_=data(774123,10_000)
    result=np.array([predict(weights,x,seq,idx)["class"] for x,seq,idx in zip(h,t,q)])
    a=actions(p,"GM",h[:128],q[:128])
    np.testing.assert_equal(result[:128],dispatch(p,h[:128],t[:128],q[:128],a).argmax(-1))
    summary={"checkpoint":str(dest.relative_to(ROOT)),"checkpoint_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
             "replayed_original_accuracy":acc,"replayed_original_nll":ce,"fresh_seed":774123,
             "fresh_n":int(y.size),"fresh_correct":int((result==y).sum()),"fresh_accuracy":float((result==y).mean()),
             "single_vs_batched_matches":128,"usage_protocol_sha256":pin}
    (OUT/"usability.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
