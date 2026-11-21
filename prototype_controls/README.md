# Independent fixed-depth 30M decoders

This supplement trains separate networks for one and two recurrent passes at deployment. The primary study has independent four-pass and adaptive networks. Forced exits measured with those primary networks use weights optimized for different objectives.

```sh
.venv/bin/python prototype_controls/run.py
.venv/bin/python prototype_controls/summarize.py
```

The fixed controls use the primary 30M backbone, byte streams, initialization, update rule and batch size. Original test scores cover the same target bytes for every architecture. The external evaluation uses the same 154-sonnet processing and windows as the independent text follow-up. The run entry point reuses completed cells and writes results atomically. It requires the two adaptive seed-11 checkpoints from the primary prototype before its replay stage. The summary also requires the primary three-seed results at 512 steps and seed-11 results at 2,048.

See `protocol.md` for the declared comparisons. See `results/prototype_controls/all_scores.csv` for model-level costs, depths, bits per byte and byte perplexity. The `paired_bpb.csv` table gives adaptive-minus-fixed differences. Negative values favor the adaptive model in predictive quality. Three-seed paired intervals describe variation from training on the single fixed play segment. The independent Sonnets comparison uses only the one exported adaptive seed, so its comparison has no seed interval. Computation estimates count dense-equivalent matrix operations and exclude the optimizer, routing synchronization and data movement. Comparable model sizes and training windows do not equalize training FLOPs across pass counts. Training times include first compilation and vary with other workloads on the M4.

The external loader was reformatted while the first fixed model trained. The next cell stopped at its source-hash guard before training. The registered loader survives byte-for-byte as `prototype_controls/sonnets.py`. Both loader versions produce identical arrays across all 1,828 windows; `results/prototype_controls/window_provenance.json` records both source hashes and both corpus window hashes. The pending cells resumed using the registered source.
