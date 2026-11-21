"""Repair pre-int64 pilot metric artifacts from retained per-example depths; assert all costs."""
import csv
import gzip
import json
import sys
from collections import defaultdict

from common import OUT, forward_flops


def audit(repair=False):
    checked = repaired = 0
    for path in sorted(OUT.glob("*-w*.json")):
        if path.name.split("-")[0] not in {"main","scale","long","opt","usable"}:
            continue
        row = json.loads(path.read_text())
        table = path.with_suffix(".csv.gz")
        with gzip.open(table, "rt") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames
            draws = list(reader)
        by_split = defaultdict(list)
        changes = 0
        for sample in draws:
            value = int(forward_flops(row["width"],int(sample["length"]),int(sample["passes"]),row["mode"] == "adaptive"))
            if value != int(sample["matmul_flops"]):
                if not repair:
                    raise ValueError(f"{table}: FLOP mismatch")
                sample["matmul_flops"] = value
                changes += 1
            by_split[sample["split"]].append(value)
        for split in row["splits"]:
            values = by_split[split["name"]]
            cost = sum(values)/len(values)
            if abs(cost - split["mean_flops_est"]) > 0.01:
                if not repair:
                    raise ValueError(f"{path}: summary FLOP mismatch")
                split["mean_flops_est"] = cost
                changes += 1
        if changes:
            with gzip.open(table, "wt") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(draws)
            path.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            print(f"Recomputed recorded cost fields from archived exits in {path.name}: {changes} fields")
            repaired += 1
        checked += len(draws)
    print(f"Audited {checked} per-example inference FLOP costs. Corrected {repaired} runs.")


if __name__ == "__main__":
    audit("--repair" in sys.argv)
