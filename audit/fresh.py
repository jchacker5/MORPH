"""Compare independent reruns using deterministic task endpoints, excluding timing."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

SPECS = [
    ("joint/fits.csv", ["seed", "cell"], ["accuracy", "cross_entropy", "mean_passes", "compact_inference_linear_macs"]),
    ("mechanisms/grouping.csv", ["seed", "mode", "slots"], ["accuracy", "logloss", "ari", "max_suffix_delta"]),
    ("mechanisms/routing.csv", ["seed", "mode"], ["accuracy", "active_params"]),
    ("mechanisms/memory.csv", ["seed", "mode", "slots", "condition", "length"], ["accuracy", "key_hit", "cache_bytes"]),
]

def table(folder,file,keys):
    with (folder/"results"/file).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(keys).issubset(fields):
            raise ValueError(f"Invalid replication header: {file}")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed replication row in {file} at line {reader.line_num}")
            if any(not row[k].strip() for k in keys):
                raise ValueError(f"Empty record key in {file} at line {reader.line_num}")
            key = tuple(row[k] for k in keys)
            if key in rows:
                raise ValueError(f"Duplicate record in {file}: {key}")
            rows[key] = row
        if not rows:
            raise ValueError(f"Empty replication table: {file}")
        return rows


def compare(original, fresh_folder, specs=SPECS):
    """Validate complete tables before returning deterministic endpoint differences."""
    results = {}
    for file, keys, metrics in specs:
        orig = table(original, file, keys)
        fresh = table(fresh_folder, file, keys)
        if orig.keys() != fresh.keys():
            raise ValueError(f"Incomplete replication for {file}")
        deltas = {}
        for metric in metrics:
            try:
                pairs = [(float(row[metric]), float(fresh[key][metric])) for key, row in orig.items()]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{file}: missing or invalid metric {metric}") from error
            if not all(math.isfinite(a) and math.isfinite(b) for a, b in pairs):
                raise ValueError(f"{file}: nonfinite metric {metric}")
            deltas[metric] = max(abs(a - b) for a, b in pairs)
        if max(deltas.values()) > 1e-6:
            raise ValueError(f"{file}: {deltas}")
        results[file] = {"rows": len(orig), "max_abs_difference": deltas}
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("fresh",type=Path)
    args=p.parse_args()
    results=compare(ROOT,args.fresh)
    (ROOT/"results/fresh_replication.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
