"""Create a portable research archive without the virtual environment or Git internals."""
from pathlib import Path
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"dist"

def skip(p):
    rel=p.relative_to(ROOT)
    return (any(part in {".git",".venv","__pycache__","dist"} for part in rel.parts)
            or rel.parts[:2]==("paper","build")
            or p.name in {".DS_Store"} or p.suffix==".pyc")

def main():
    OUT.mkdir(exist_ok=True)
    target=OUT/"MORPH-replication.zip"
    tmp=OUT/"MORPH-replication.zip.partial"
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for name in sorted(ROOT.iterdir()):
            if skip(name): continue
            files=name.rglob("*") if name.is_dir() else (name,)
            for f in sorted(files):
                if f.is_file() and not skip(f):
                    z.write(f,Path("MORPH")/f.relative_to(ROOT))
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad: raise ValueError(f"Archive verification failed: {bad}")
        expected=["MORPH/paper/MORPH.pdf","MORPH/results/verification.json",
                  "MORPH/results/lm/prototype/adaptive-11-2048.safetensors","MORPH/results/lm_followup/capstone.safetensors"]
        assert all(name in z.namelist() for name in expected)
        members=len(z.infolist())
    tmp.replace(target)
    with target.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    (OUT/"SHA256SUMS").write_text(f"{digest}  {target.name}\n")
    print(f"{members} verified files; {target.stat().st_size:,} bytes; SHA-256 {digest}")
if __name__=="__main__": main()
