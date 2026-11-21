"""Validate the archived matrices and reproducible primary numerical contrasts."""
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results"

def load(path): return json.loads(path.read_text())

def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def primary_joint():
    with (R/"joint/fits.csv").open() as f: rows=list(csv.DictReader(f))
    cells={s:{r["cell"]:float(r["accuracy"]) for r in rows if int(r["seed"])==s} for s in (41,42,43)}
    assert all(len(c)==17 for c in cells.values())
    values=[100*(c["DGRM"]-sum(c[k] for k in "DGRM")+3*c["fixed"]) for c in cells.values()]
    with (R/"joint/contrasts_pp.csv").open() as f: reference=next(csv.DictReader(f))
    assert np.isclose(values,[float(reference[f"seed{s}"]) for s in cells]).all()
    return float(np.mean(values))

def main():
    counts={}
    for prefix,expected in [("main",21),("scale",18),("opt",21),("long",45),("usable",2)]:
        paths=list((R/"reasoning").glob(f"{prefix}-w*-s*.json"))
        assert len(paths)==expected,(prefix,len(paths))
        counts["reasoning_"+prefix]=len(paths)
    for label,n in [("grouping",42),("routing",12),("memory",180),("memory_extension",45)]:
        rows=load(R/f"mechanisms/{label}.json")
        assert len(rows)==n,(label,len(rows))
        assert all(0<=x["accuracy"]<=1 for x in rows)
        counts[label+"_condition_rows"]=len(rows)
    assert len(list((R/"lm/final").glob("*.json")))==18
    for file,expected in [("lm/prototype",8),("prototype_controls",8)]:
        glob="fixed[12]-*-*.json" if file=="prototype_controls" else "*-*-*.json"
        paths=list((R/file).glob(glob))
        assert len(paths)==expected,(file,len(paths))
    for prefix,n in [("long",12),("replay",4)]:
        paths=list((R/"lm_followup").glob(prefix+"-*.json"))
        assert len(paths)==n,(prefix,len(paths))
    search=sorted((R/"lm/search").glob("trial-*.decision.json"))
    assert [load(p)["status"] for p in search]==["keep","discard","discard","discard"]
    assert load(R/"lm/frozen.json")==load(search[0])
    v=load(R/"lm/verification.json")
    assert len(v["checkpoints"])==2 and len(set(v["mixed_reordering_exits"]))==4
    cap=load(R/"lm_followup/capstone.json")
    assert cap["seconds"]>=300
    check=load(R/"joint/usability.json")
    assert sha(R/"joint/gm_seed41.npz")==check["checkpoint_sha256"] and check["fresh_correct"]==9999
    fresh=load(R/"fresh_replication.json")
    assert sum(r["rows"] for r in fresh.values())==285
    # Keep a byte-exact inventory of deliverable weights independent of Git's storage format.
    weights={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(R.rglob("*")) if p.suffix in (".safetensors",".npz")}
    report=dict(matrix_counts=counts,primary_joint_contrast_pp=primary_joint(),machine=mx.device_info(),os=platform.platform(),
                python=platform.python_version(),weights=weights)
    (R/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Verified all expected matrices, {len(weights)} saved weight files and 285 independently replayed rows.")
if __name__=="__main__": main()
