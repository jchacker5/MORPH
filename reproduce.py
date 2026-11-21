"""Stage source, data and protocol pins in a fresh directory for independent reruns."""
import argparse
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
SUITES=("lm","lm_followup","mechanisms","joint","reasoning","prototype_controls")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    dest=parser.parse_args().destination.expanduser().resolve()
    if dest.exists() or dest.is_relative_to(ROOT):
        raise SystemExit("Use a new directory outside the archived study.")
    dest.mkdir(parents=True)
    shutil.copy2(ROOT/"requirements.txt",dest)
    for folder in (*SUITES,"docs"):
        shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for method in ("lm","lm_followup","mechanisms","joint","reasoning","prototype_controls"):
        src=ROOT/"results"/method
        out=dest/"results"/method
        out.mkdir(parents=True)
        for pattern in ("*.sha256","*freeze*.txt","*registered*.txt","*frozen*.txt"):
            for f in src.glob(pattern): shutil.copy2(f,out/f.name)
    out=dest/"results/lm"
    (out/"final").mkdir()
    (out/"prototype").mkdir()
    # Keep the originally selected validation plan as an immutable input to confirmation.
    shutil.copy2(ROOT/"results/lm/frozen.json",out)
    shutil.copytree(ROOT/"results/lm/search",out/"search")
    print(f"Fresh study staged at {dest}. Select a reproduction command from the project README.")
if __name__=="__main__": main()
