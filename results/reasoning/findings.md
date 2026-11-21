# MORPH Boolean reasoning study

Joseph Defendre

Experiments 1, 6, 7, and 8. Each primary comparison uses three paired training seeds and identical test draws within a study. Intervals are descriptive two-sided 95% Student-t intervals for the mean over seeds, conditional on the locked test sample. Every quality percentage is exact final-bit accuracy.

## Matched-compute reasoning results

The initial study used 120 updates per fit. The registered follow-up used 2,400 updates on fresh streams after the pilot suggested undertraining. Each row in the following table represents three independent fits in each study. Costs use theoretical forward matrix products. Data generation and optimizer steps were matched within each comparison.

| Inference policy | Parameters | Pilot accuracy (%) | Longer accuracy (%) | Longer MFLOPs/example | Longer passes |
|---|---:|---:|---:|---:|---:|
| Shared fixed 1 | 50,498 | 50.35 | 93.52 | 9.18 | 1.00 |
| Shared fixed 2 | 50,498 | 50.46 | 96.88 | 18.35 | 2.00 |
| Shared fixed 3 | 50,498 | 50.46 | 96.53 | 27.53 | 3.00 |
| Shared fixed 4 | 50,498 | 50.46 | 96.99 | 36.70 | 4.00 |
| Untied fixed 4 | 50,018 | 57.18 | 98.78 | 11.01 | 4.00 |
| Random depth | 50,498 | 50.46 | 95.66 | 23.30 | 2.53 |
| MORPH | 50,563 | 50.46 | 85.53 | 23.38 | 2.60 |

Pilot. MORPH minus the best tested fixed policy or fixed mixture within its FLOP budget was -6.71 [-29.88, 16.45] percentage points. The strict equal-cost mixture gave -4.81 [-21.40, 11.78] points. The fixed envelope is chosen from the observed test means for each seed and is therefore an optimistic comparator. The gain over fixed-four under closely matched training cost was 0.00 [0.00, 0.00] points; relative training-cost difference was 0.00319%.

The uncalibrated 0.5 policy used 1.00 mean passes. The development-calibrated policy used 1.99. Compared with 32 length-stratified exit permutations it gained 0.00 [0.00, 0.00] points. Seed-wise instruction-count/depth Spearman coefficients were [0.04512863974184127, -0.0004731668852202928, 0.020452750419921152]. Forced depths 1–4 gave 50.46%, 50.46%, 50.46%, 50.46%. The archived rows include actual hard exits, forced predictions, and controller probabilities.

Longer study. MORPH minus the best tested fixed policy or fixed mixture within its FLOP budget was -13.36 [-34.94, 8.22] percentage points. The strict equal-cost mixture gave -12.57 [-35.73, 10.60] points. The fixed envelope is chosen from the observed test means for each seed and is therefore an optimistic comparator. The gain over fixed-four under closely matched training cost was -11.46 [-36.84, 13.92] points; relative training-cost difference was 0.00313%.

The uncalibrated 0.5 policy used 1.21 mean passes. The development-calibrated policy used 2.60. Compared with 32 length-stratified exit permutations it gained -2.33 [-19.96, 15.30] points. Seed-wise instruction-count/depth Spearman coefficients were [-0.1163995477443302, -0.019216584433369734, -0.0711875081216866]. Forced depths 1–4 gave 89.24%, 91.38%, 91.03%, 82.64%. The archived rows include actual hard exits, forced predictions, and controller probabilities.

## Generalization after longer training

| Split | MORPH (%) | Shared fixed four (%) | MORPH passes | MORPH vs fixed frontier (pp, 95% seed interval) |
|---|---:|---:|---:|---:|
| id | 85.53 | 96.99 | 2.60 | -13.36 [-34.94, 8.22] |
| id64 | 83.33 | 95.83 | 2.53 | -14.57 [-34.63, 5.48] |
| length256 | 70.14 | 70.83 | 2.03 | -8.33 [-19.61, 2.95] |
| length512 | 67.01 | 73.96 | 1.78 | -5.62 [-15.51, 4.27] |
| length1024 | 61.46 | 62.85 | 1.54 | 1.53 [-8.51, 11.57] |
| composition64 | 84.03 | 92.36 | 2.50 | -10.07 [-17.97, -2.16] |

The ID pool has 576 examples, 192 at each of 32, 64 and 128 tokens. Shift rows contain 96 examples each. The composition row has 4–6 instructions at length 64, while id64 has 1–3 instructions at that same length. Negative generalization results must be interpreted within this Boolean execution task.

## Scaling synthetic reasoning

| MORPH parameters | Pilot adaptive/fixed-four (%) | Longer adaptive (%) | Longer fixed 1/2/3/4 (%) | Adaptive gain vs fixed envelope (pp, 95% seed interval) | Training TFLOPs/fit, MORPH |
|---:|---:|---:|---:|---:|---:|
| 9,653,507 | 46.88/46.88 | 67.19 | 63.54 / 67.71 / 69.40 / 60.42 | 3.65 [-30.00, 37.29] | 9.527 |
| 30,755,203 | 51.04/48.96 | 67.19 | 63.28 / 59.64 / 62.89 / 56.64 | 3.91 [-1.23, 9.04] | 30.300 |
| 99,596,163 | 50.00/48.96 | 62.37 | 68.10 / 65.23 / 72.01 / 60.42 | -5.73 [-33.81, 22.35] | 98.026 |

The pilot had 16 updates and 1,024 training tokens per fit. The longer replication had 160 updates and 40,960 training tokens per fit across all 45 models. Its data budget is too small to infer an asymptotic scaling law. Width, active parameters and FLOPs change together across sizes. Within each size the trained policies saw the same input tokens. Each adaptive model trains all four exits and is compared independently with a parameter- and training-FLOP-matched shared fixed-four model and an inference-cost frontier.

## Engineering qualification

Two fresh seed-17 models trained for 8,000 updates on identical data, then ran on a separate qualification split. Pooled ID qualification requires 95% observed accuracy and a 93% Wilson lower bound. Shift qualification also requires at least 90% in each shift.

| Checkpoint | Correct / ID examples | ID accuracy (Wilson 95%) | ID qualified | Shift-qualified | Accuracy at lengths 256 / 512 / 1,024 / composition (%) |
|---|---:|---:|---|---|---:|
| MORPH | 1087 / 1152 | 94.36 [92.87, 95.55] | False | False | 76.56 / 59.90 / 56.25 / 83.33 |
| Shared fixed 2 | 1136 / 1152 | 98.61 [97.76, 99.14] | True | False | 90.10 / 71.35 / 65.62 / 89.58 |

The four evaluation shifts are assessed per checkpoint. These are runnable binary program classifiers over the generated task grammar. General-purpose or conversational performance was outside this reasoning benchmark.

## Reproduce

Run the following from the standalone project root after installing the locked dependencies.

```sh
.venv/bin/python reasoning/suite.py all
.venv/bin/python reasoning/extended.py
.venv/bin/python reasoning/optimize.py
.venv/bin/python reasoning/qualify.py
.venv/bin/python reasoning/verify.py --full
for study in main scale opt long; do .venv/bin/python reasoning/summarize.py "$study"; done
.venv/bin/python reasoning/report.py
.venv/bin/python reasoning/predict.py 1 not zero not --checkpoint adaptive --length 64
```

The per-example compressed CSV files, raw per-run JSON, generated metrics tables, train-step times and theoretical FLOP estimates accompany this supplement. Calibrated hard exits physically compact the batch; the cached full-depth analysis permits interventions. Machine timings share the local accelerator with other work. MLX peak allocation is process-scoped. Data and optimizer schedules differ between the pilot and registered follow-ups; their differences are descriptive, not an isolated treatment effect.

## Exploratory halt-threshold sensitivity

The prespecified primary search uses a 0.001-spaced grid. A post hoc search over every development hazard breakpoint isolated whether this spacing masked input dependence. Both policies use the same development data, 2.5-pass objective and lowest-threshold tie rule. This sensitivity reuses the original held-out examples.

| Width | Mean calibrated test passes | Refined accuracy (%) | Gain vs fixed frontier (pp, 95% seed interval) | Shuffle gain (pp) |
|---|---:|---:|---:|---:|
| opt-64 | 2.53 | 92.59 | -6.27 [-18.73, 6.19] | 2.29 |
| long-896 | 2.48 | 67.06 | -5.75 [-14.48, 2.99] | -0.53 |
| long-1600 | 2.17 | 70.05 | 1.88 [-2.91, 6.68] | 1.45 |
| long-2880 | 2.37 | 62.63 | -8.65 [-45.98, 28.68] | 0.14 |

The companion sensitivity JSON quantifies replay and compact-versus-cached disagreements. When original scaled weights had been discarded, replay under the recorded seed sometimes produced different predictions. Within-replay comparisons hold the weights fixed, and a second cached check recalibrates using the first half of archived original test hazards at each length and scores the other half against original fixed baselines. Both are post hoc checks on reused test data.

| Width | Original-weight second-half passes | Second-half accuracy (%) | Second-half gain vs fixed envelope (pp) | Within-replay coarse-to-refined accuracy change (pp) |
|---|---:|---:|---:|---:|
| opt-64 | 2.54 | 90.28 | -8.73 [-16.67, -0.78] | -0.29 |
| long-896 | 2.42 | 61.98 | -6.12 [-11.62, -0.63] | -0.52 |
| long-1600 | 2.31 | 61.46 | -2.97 [-7.27, 1.33] | 3.52 |
| long-2880 | 2.38 | 58.07 | -7.78 [-40.01, 24.45] | 0.26 |

All scientific study rows above use 21 original pilot fits, 18 original scale fits, 21 extended small fits, 45 extended scale fits and two single-seed engineering qualifications. Profiling and sensitivity replays are separate.
