# Isolated mechanism probes

Joseph Defendre · MORPH · November 21, 2026

These experiments ran from scratch on three paired CPU seeds, 11, 29, and 47. Each test condition contained 2,048 independently generated queries per seed. The protocol digest is recorded next to the raw JSON and CSV. Intervals below use paired per-seed differences and a two-sided 95% Student t interval with two degrees of freedom. Endpoints are sensitive to the small number of seeds. Model selection across a set of comparators does not receive a multiplicity adjustment. These component probes are separate from the joint model.

### Experiment 2 · Grouping

| Record slots | Learned PPL | Random symbol map | Adjacent segments | Fixed stride | Learned ARI |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.915 | 2.244 | 2.417 | 2.413 | 0.382 |
| 4 | 1.156 | 1.689 | 2.013 | 2.000 | 0.814 |
| 8 | 1.038 | 1.183 | 1.528 | 1.528 | 1.000 |

The predictive metric is query-value conditional perplexity averaged across seeds. At four slots, learned assignments improved log loss by 0.377 nats relative to the strongest compressed control. Across seeds, learned minus control was −0.569, −0.241, and −0.320 nats and the 95% interval was [−0.802, +0.048]. A family oracle obtained 1.044 PPL at four slots. An uncompressed 16-slot learned reader obtained 1.059. At eight slots, perfect ARI means the model used a partition equivalent to the four latent families over observed records; the module still allocated eight slots. Permuting the suffix, including the target token, left predictions identical in every arm.

The predictor used only observations to the left of the query target. Prefixes had sixteen records, consisting of one symbol and one value each. The learned four-slot cache stored histograms of twelve keys and four values in float32, 256 bytes in total, versus 1,024 bytes for the same sixteen-slot representation. A packed raw list of sixteen byte-sized symbol/value pairs would occupy 32 bytes. Slot compression alone provides no practical byte advantage over this alternative.

### Experiment 3 · Expert routing and intervention

| Policy | Exact match | Active parameters | Total parameters | Task–expert MI, bits | Forced cyclic route |
|---|---:|---:|---:|---:|---:|
| Learned expected risk | 100.00% | 632 | 2,300 | 0.234 | 84.15% |
| Conventional top one | 99.80% | 632 | 2,300 | 0.123 | 33.04% |
| Uniform random | 100.00% | 556 | 2,224 | 0.004 | 100.00% |
| Dense | 100.00% | 2,224 | 2,224 | — | — |

The paired learned-minus-conventional accuracy difference was +0.20 percentage points [−0.65, +1.04]. Forcing each learned route to its next numbered expert reduced accuracy for every seed; the mean loss was 15.85 points [−5.62, +37.32]. This cyclic permutation provides one intervention per seed and depends on arbitrary expert numbering. Uniform random routing equaled learned routing at a lower active parameter count. All trainers materialized all four expert outputs. The sparse prediction path dispatched only selected experts. These finite-support tasks admit near-perfect interpolation. Task markers explicitly identify the type of operation, and every expert sees the complete input. The evidence does not establish a need for specialists.

### Experiment 4 · Memory

| Policy, four-slot limit | IID recall | Joint shift, length 32 | Joint shift, length 96 | Retained array bytes at length 32 | Median-block CPU µs/query at length 32, joint shift |
|---|---:|---:|---:|---:|---:|
| Full context | 100.0% | 100.0% | 100.0% | 96 | 6.58 |
| Last four | 17.68% | 6.28% | 6.84% | 12 | 6.74 |
| Uniform stride | 17.58% | 18.26% | 10.95% | 12 | 2.77 |
| Learned retention | 100.0% | 100.0% | 100.0% | 28 | 69.11 |
| Permuted learned priority | 17.01% | 18.15% | 10.74% | 28 | 84.79 |
| Explicit cue rule | 100.0% | 100.0% | 100.0% | 12 | 3.90 |

At length 32 with cue frequencies and relevant positions shifted, learned retention gained 81.74 points [79.62, 83.85] against the strongest generic capacity-matched comparator. Permuting its priorities cost 81.85 points [81.25, 82.45]. The handcrafted cue rule matched learned accuracy with fewer cache bytes and lower latency. The learned selector has 325 float32 parameters outside its 28-byte cache. Cache bytes exclude input, model, temporaries and Python overhead. The reported latency is the mean across seeds of each seed's median over three 48-query NumPy timing blocks. These times are environment-specific and include record processing and lookup. The memory problem supplies a cue before the stream and exactly four relevant records. Dense cue/color outer-product features make a four-color lookup table easy to learn. The held-out shift uses colors seen infrequently during training. Full storage performs exact key lookup, not dense Transformer attention.

### Reproduction and bounded use

From the project root run each command with the project's pinned environment.

```bash
.venv/bin/python -m mechanisms.grouping
.venv/bin/python -m mechanisms.routing
.venv/bin/python -m mechanisms.memory
.venv/bin/python -m mechanisms.analyze
.venv/bin/python -m mechanisms.export_recall
```

The saved checkpoint is results/mechanisms/recall-seed11-k4.npz. Replay reproduced every retained seed-11 four-slot accuracy and hit rate in all five conditions. The following invocation returns value 10 for key 15, with a cache hit and 28 retained array bytes.

```bash
echo '{"cue":2,"records":[[11,7,2],[12,3,1],[13,9,0],[14,6,2],[15,10,2],[16,4,2],[17,14,1],[18,8,3]],"query":15}' | .venv/bin/python -m mechanisms.recall
```

Each record is [unique key, value, color]. Supply cue and color in 0–3, value in 0–15, and a unique unsigned 16-bit key. The learned policy uses the declared episode length to normalize position. Usability evidence covers the generated recall task with 32 or 96 records and four relevant keys. The explicit cue rule also solves that task exactly.

### Exploratory extension · Dense attention and long-stream recall

After examining the original registered results, we froze a supplemental recipe and drew 2,048 new double-shift episodes per seed at each of lengths 32, 96, and 1,024. Full dense single-query softmax attention, the learned four-slot cache and the four-slot cue rule all had 100% exact recall at every length. At 1,024 records, dense attention retained 3,072 bytes of packed keys and values, the learned policy retained 28 bytes, and the cue rule retained 12 bytes. Temporary attention tensors and model parameters are additional. The direct-mapped four-slot hash cache had 6.20% accuracy, near the 6.25% deterministic-zero fallback expectation on uniform values. Reference latency on the longest sequences averaged 66.45 µs for vectorized full attention, 2,704.41 µs for the Python record-by-record learned policy, and 59.92 µs for the cue rule. These measurements compare the provided implementations and do not isolate the effect of architecture. The generator still supplies exactly four relevant records regardless of length. Reproduce the 45 rows in memory_extension.json with `.venv/bin/python -m mechanisms.recall_extension`.
