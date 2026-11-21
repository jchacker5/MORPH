"""Apply the frozen K=4 learned memory probe to a JSON input on stdin."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .common import OUT
from .memory import reference_stream


def predict(request, weights):
    if not isinstance(request, dict):
        raise TypeError("request must be an object")
    cue = request.get("cue")
    query = request.get("query")
    records = request.get("records")
    if type(cue) is not int or not 0 <= cue < 4:
        raise ValueError("cue must be an integer from 0 to 3")
    if type(query) is not int or not 0 <= query <= 65535:
        raise ValueError("query must be an integer from 0 to 65535")
    if not isinstance(records, list) or not 4 <= len(records) <= 65536:
        raise ValueError("supply 4–65536 records as [key,value,color] triples")
    if any(
        not isinstance(rec, list)
        or len(rec) != 3
        or any(type(v) is not int for v in rec)
        or not (0 <= rec[0] <= 65535 and 0 <= rec[1] < 16 and 0 <= rec[2] < 4)
        for rec in records
    ):
        raise ValueError(
            "each record needs an integer key 0–65535, value 0–15 and color 0–3"
        )
    if len({r[0] for r in records}) != len(records):
        raise ValueError("record keys must be unique")
    keys, values, colors = np.asarray(records).T
    ans, hit, cache = reference_stream(
        weights, "learned", cue, colors, keys, values, query, 4
    )
    return {"value": ans, "found_in_cache": hit, "retained_array_bytes": cache}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=OUT / "recall-seed11-k4.npz")
    args = parser.parse_args()
    try:
        with np.load(args.weights, allow_pickle=False) as weights_file:
            weights = {k: weights_file[k] for k in weights_file.files}
        print(json.dumps(predict(json.load(sys.stdin), weights)))
    except (ValueError, KeyError, TypeError) as e:
        parser.exit(2, f"Invalid input: {e}\n")
