"""Verify full grids and compute registered paired effects from retained rows."""

import json
from itertools import product

import numpy as np

from .common import N_TEST, OUT, SEEDS, interval, verify_registration
from .memory import CONDITIONS, MODES


def load(name, dimensions, expected):
    rows = json.loads((OUT / (name + ".json")).read_text())
    keys = [tuple(r[d] for d in dimensions) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) == set(expected)
    digest = verify_registration()
    assert all(
        r["protocol_sha256"] == digest and r["test_examples"] == N_TEST for r in rows
    )
    return {key: row for key, row in zip(keys, rows)}


def main():
    compressed = ("learned", "random", "adjacent", "stride")
    groupkeys = list(product(SEEDS, (2, 4, 8), compressed))
    groupkeys.extend(
        (s, k, m) for s in SEEDS for k, m in ((4, "oracle"), (16, "uncompressed"))
    )
    groups = load("grouping", ("seed", "slots", "mode"), groupkeys)
    routes = load(
        "routing",
        ("seed", "mode"),
        product(SEEDS, ("risk", "switch", "random", "dense")),
    )
    mem = load(
        "memory",
        ("seed", "slots", "condition", "mode"),
        product(SEEDS, (4, 8), CONDITIONS, MODES),
    )
    assert all(r["max_suffix_delta"] == 0 for r in groups.values())
    assert all(0 <= r["key_hit"] <= r["accuracy"] <= 1 for r in mem.values())
    gbase = min(
        compressed[1:], key=lambda m: sum(groups[s, 4, m]["logloss"] for s in SEEDS)
    )
    mbase = max(
        ("window", "stride"),
        key=lambda m: sum(mem[s, 4, "both", m]["accuracy"] for s in SEEDS),
    )
    summary = {
        "protocol_sha256": verify_registration(),
        "rows": {"grouping": len(groups), "routing": len(routes), "memory": len(mem)},
        "seeds": SEEDS,
        "grouping_strongest": gbase,
        "grouping_logloss_learned_minus_baseline": interval(
            [
                groups[s, 4, "learned"]["logloss"] - groups[s, 4, gbase]["logloss"]
                for s in SEEDS
            ]
        ),
        "routing_accuracy_risk_minus_switch": interval(
            [
                routes[s, "risk"]["accuracy"] - routes[s, "switch"]["accuracy"]
                for s in SEEDS
            ]
        ),
        "routing_accuracy_risk_minus_random": interval(
            [
                routes[s, "risk"]["accuracy"] - routes[s, "random"]["accuracy"]
                for s in SEEDS
            ]
        ),
        "risk_accuracy_natural_minus_cyclic": interval(
            [
                routes[s, "risk"]["accuracy"] - routes[s, "risk"]["forced_accuracy"]
                for s in SEEDS
            ]
        ),
        "memory_strongest_nonoracle": mbase,
        "memory_accuracy_learned_minus_baseline": interval(
            [
                mem[s, 4, "both", "learned"]["accuracy"]
                - mem[s, 4, "both", mbase]["accuracy"]
                for s in SEEDS
            ]
        ),
        "memory_accuracy_learned_minus_permuted": interval(
            [
                mem[s, 4, "both", "learned"]["accuracy"]
                - mem[s, 4, "both", "permuted"]["accuracy"]
                for s in SEEDS
            ]
        ),
    }
    # Dependence of each routing matrix is label-invariant, including across seeds.
    from itertools import combinations

    from scipy.optimize import linear_sum_assignment

    def distance(a, b):
        ma = np.array(a)
        mb = np.array(b)
        cost = np.abs(ma[:, :, None] - mb[:, None, :]).sum(axis=0)
        ir, ic = linear_sum_assignment(cost)
        return float(cost[ir, ic].sum() / 8)

    summary[
        "routing_exploratory_mean_task_conditional_total_variation_after_best_expert_relabeling"
    ] = {
        m: [
            distance(
                routes[s, m]["task_expert_matrix"], routes[u, m]["task_expert_matrix"]
            )
            for s, u in combinations(SEEDS, 2)
        ]
        for m in ("risk", "random", "switch")
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print("Verified", summary["rows"], "and registered seed-level effects.")


if __name__ == "__main__":
    main()
