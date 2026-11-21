"""Verify follow-up budget/accounting and assemble the held-out evidence."""

import csv
import json

import numpy as np
from scipy.stats import t

from lm_followup.run import OUT, ROOT, SEEDS, digest, registration, write


def read(name):
    row = json.loads((OUT / name).read_text())
    assert row["protocol_sha256"] == registration()
    return row


def paired(a, b):
    delta = np.asarray(a) - np.asarray(b)
    half = t.ppf(0.975, len(delta) - 1) * np.std(delta, ddof=1) / np.sqrt(len(delta))
    return {
        "per_seed": delta.tolist(),
        "mean": float(delta.mean()),
        "low": float(delta.mean() - half),
        "high": float(delta.mean() + half),
    }


def summarize():
    counts = read("counts.json")
    trials = [read(f"replay-{i}.json") for i in range(1, 5)]
    assert len({r["input_bytes"] for r in trials}) == 1
    assert all(r["steps"] == 2048 and r["seed"] == 7 for r in trials)
    original = [
        json.loads((ROOT / f"results/lm/search/trial-{i:02d}.json").read_text())
        for i in range(1, 5)
    ]
    selected = json.loads((ROOT / "results/lm/frozen.json").read_text())
    rows = [
        read(f"long-{m}-{s}.json") for s in SEEDS for m in ("1", "2", "4", "adaptive")
    ]
    cap = read("capstone.json")
    assert cap["seconds"] >= 300 and cap["seed"] == 7
    assert len({r["scored_train_bytes"] for r in rows}) == 1
    assert all(
        r["steps"] == 4096
        and r["sonnets"]["examples"] == 1828
        and r["repeat_test"]["examples"] == 2323
        for r in rows
    )
    for r in [
        cap,
        next(v for v in rows if v["mode"] == "adaptive" and v["seed"] == 11),
    ]:
        cp = r["checkpoint_verification"]
        assert cp["sha256"] == digest(OUT / cp["weights_file"])
        assert (
            cp["equal_reloaded_depths"]
            and cp["generation_repeat_identical"]
            and cp["max_abs_logit_reload_delta"] == 0
        )
    modes = ("1", "2", "4", "adaptive")
    grouped = {m: [r for r in rows if r["mode"] == m] for m in modes}

    def extract(m, key, a):
        return [r[key][a] for r in grouped[m]]

    external_mean = {m: float(np.mean(extract(m, "sonnets", "bpb"))) for m in modes}
    full_rows = []
    for r in rows + [cap]:
        full_rows.append(
            {
                "seed": r["seed"],
                "mode": r["mode"],
                "stage": r["stage"],
                "steps": r["steps"],
                "parameters": r["parameters"],
                "training_bytes": r["input_bytes"],
                "training_scored_bytes": r["scored_train_bytes"],
                "training_flops": r["train_flops"],
                "train_seconds": r["seconds"],
                "training_peak_mlx_bytes": r["training_peak_mlx_bytes"],
                "external_bpb": r["sonnets"]["bpb"],
                "repeat_bpb": r["repeat_test"]["bpb"],
                "external_mean_depth": r["sonnets"]["mean_depth"],
                "external_inference_flops_per_window": r["sonnets"][
                    "inference_flops_per_window"
                ],
                "batch1_median_ms": r["batch1_median_ms"],
            }
        )
    with (OUT / "models.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=full_rows[0])
        writer.writeheader()
        writer.writerows(full_rows)
    metrics = {
        "primary_adaptive_minus_fixed2_external_bpb": paired(
            extract("adaptive", "sonnets", "bpb"), extract("2", "sonnets", "bpb")
        ),
        "adaptive_minus_fixed1_external_bpb": paired(
            extract("adaptive", "sonnets", "bpb"), extract("1", "sonnets", "bpb")
        ),
        "adaptive_minus_fixed4_external_bpb": paired(
            extract("adaptive", "sonnets", "bpb"), extract("4", "sonnets", "bpb")
        ),
        "adaptive_minus_bigram_external_bpb": paired(
            extract("adaptive", "sonnets", "bpb"), [counts["sonnets"]["bpb"]["1"]] * 3
        ),
        "adaptive_minus_order3_external_bpb": paired(
            extract("adaptive", "sonnets", "bpb"), [counts["sonnets"]["bpb"]["3"]] * 3
        ),
        "replay_objective": [r["validation_objective"] for r in trials],
        "replay_best_trial": int(
            np.argmin([r["validation_objective"] for r in trials]) + 1
        ),
        "original_selected_trial": selected["trial"],
        "mean_external_bpb": external_mean,
        "protocol_sha256": registration(),
        "paired_seeds": SEEDS,
    }
    write(OUT / "summary.json", metrics)
    out = [
        "# Small language model follow-up",
        "",
        "This follow-up used a configuration frozen after the earlier time-limited search. It held update counts constant for the replay and strengthened the paired comparison with 4,096 updates per model. All four recurrent architectures saw the same training windows within each seed. The frozen three-seed primary outcome was bits per byte on the 154 sonnets. The external test contained 87,744 scored bytes, with no exact 65-byte test window in the training corpus. The source, cleaned data hash and extraction code accompany the protocol. [Project Gutenberg edition](https://www.gutenberg.org/ebooks/1041) · [Pinned source](https://github.com/GITenberg/Shakespeare-s-Sonnets_1041/blob/b6da8a3e73e7081cf71c57e930001d0b92a8f644/1041.txt)",
        "",
        "## Matched-update validation replay",
        "",
        "| Trial | Original 30 s steps | Original penalized BPB | Replay steps | Replay BPB | Replay mean passes | Replay penalized BPB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (old, new) in enumerate(zip(original, trials), 1):
        ob = old["val"]["bpb"] + 0.08 * max(0, old["val"]["mean_depth"] - 2)
        out.append(
            f"| {i} | {old['steps']} | {ob:.4f} | {new['steps']} | {new['validation']['bpb']:.4f} | {new['validation']['mean_depth']:.3f} | {new['validation_objective']:.4f} |"
        )
    out.extend(
        [
            "",
            f"The original protocol selected trial {selected['trial']}. Trial {metrics['replay_best_trial']} scored lowest in the equal-update replay. All longer runs used the original selection.",
            "",
            "## Three-seed paired experiment",
            "",
            "| Pass policy | Parameters | External BPB | Repeat test BPB | External mean passes | Train TFLOPs | External inference MFLOPs/window | Batch-one ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m in modes:
        rr = grouped[m]
        out.append(
            f"| {m} | {rr[0]['parameters']:,} | {external_mean[m]:.4f} | {np.mean(extract(m, 'repeat_test', 'bpb')):.4f} | {np.mean(extract(m, 'sonnets', 'mean_depth')):.3f} | {np.mean([r['train_flops'] for r in rr]) / 1e12:.3f} | {np.mean(extract(m, 'sonnets', 'inference_flops_per_window')) / 1e6:.3f} | {np.mean([r['batch1_median_ms'] for r in rr]):.3f} |"
        )
    c = metrics["primary_adaptive_minus_fixed2_external_bpb"]
    out.extend(
        [
            "",
            f"Adaptive minus fixed two-pass external BPB averaged {c['mean']:+.4f}, with a 95% paired Student interval [{c['low']:+.4f}, {c['high']:+.4f}]. Per-seed differences were {', '.join(format(x, '+.4f') for x in c['per_seed'])}. Intervals have two degrees of freedom. The neural runs matched updates and input bytes. Four adaptive training exits incur extra FLOPs.",
            "",
            f"Count models trained on exactly {counts['training_bytes']:,} Tiny Shakespeare bytes gave {counts['sonnets']['bpb']['1']:.4f} BPB (bigram) and {counts['sonnets']['bpb']['3']:.4f} BPB (three-byte history) on the sonnets. Their repeat-test scores were {counts['repeat_test']['bpb']['1']:.4f} and {counts['repeat_test']['bpb']['3']:.4f}. Each reported latency averages three seeds' median batch-one times across 40 synchronized windows with shared machine load. FLOP estimates count dense matrix arithmetic and exclude scalar kernels, data movement and optimizer updates.",
            "",
            "## Five-minute capstone",
            "",
            f"Seed 7 completed {cap['steps']:,} optimizer steps and consumed {cap['input_bytes']:,} input bytes in {cap['seconds']:.3f} seconds. Validation BPB was {cap['validation']['bpb']:.4f}, external BPB {cap['sonnets']['bpb']:.4f}, and repeated test BPB {cap['repeat_test']['bpb']:.4f}. External mean depth was {cap['sonnets']['mean_depth']:.3f}. This single shared-hardware duration run uses the earlier development seed.",
            "",
            "## Frozen inference examples",
            "",
            "Both retained models reproduced all eight checked logits/depths on reload and identical bytes on two generation calls with the registered prompt, seed and temperature. Each sample contains the original prompt followed by 160 sampled bytes. The raw hex bytes are in the per-model JSON.",
            "",
        ]
    )
    for title, r in (
        ("4,096-step seed 11", grouped["adaptive"][0]),
        ("300-second seed 7", cap),
    ):
        out.extend(
            [
                title,
                "",
                "~~~~text",
                r["checkpoint_verification"]["sample_utf8"],
                "~~~~",
                "",
            ]
        )
    out.extend(
        [
            "## Reproduction",
            "",
            "Run from the project root with the study environment. The corpus is included and its digest is checked before use.",
            "",
            "~~~sh",
            ".venv/bin/python -m lm_followup.run counts",
            ".venv/bin/python -m lm_followup.run replay",
            ".venv/bin/python -m lm_followup.run long",
            ".venv/bin/python -m lm_followup.run capstone",
            ".venv/bin/python -m lm_followup.analyze",
            '.venv/bin/python lm/generate.py --small --weights results/lm_followup/long-adaptive-11.safetensors --prompt "To be, or not to be" --seed 73 --temperature .8 --bytes 160',
            "~~~",
            "",
            "The five-minute run depends on system throughput. Existing completed cells are reused. Use a fresh output directory in an isolated project copy for an independent replication and retain the protocol and digest. The sonnet scores measure related-author literary next-byte prediction. They do not evaluate instruction following, factuality, or useful answers to questions.",
        ]
    )
    (OUT / "findings.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    summarize()
