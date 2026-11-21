# MORPH joint-controller experiment

Joseph Defendre

On a four-class synthetic retrieval task, learning which memory bank to keep and how to pair its contents gave 100% accuracy for all three seeds on 4,096 held-out examples. All four learned decisions together averaged 91.17%. The full system beat the fixed factorial baseline by 55.66 percentage points. Evidence for four-way benefit remained inconclusive under the registered superadditivity test. The task intentionally makes memory and grouping informative.

## Reproduction

Run from the MORPH project root with its pinned Python environment. The NumPy inference CLI needs only NumPy and the GM checkpoint. MLX on CPU, SciPy and Matplotlib reproduce the fitting and analysis.

```sh
.venv/bin/python joint/run.py --verify-only
.venv/bin/python joint/run.py
.venv/bin/python joint/analyze.py
.venv/bin/python joint/replay.py
.venv/bin/python joint/static_control.py
.venv/bin/python joint/infer.py --header 2 0 0 0 --tokens 0 1 2 3 2 0 3 1 --query 1
```

The final command predicts class 1. It retains the later bank, forms interleaved pairs, selects expert 1 and executes two recurrent steps. The header and symbol encoding are part of this benchmark's input contract. The GM model processes one example at a time. The full learned-depth configuration uses a cohort of 64 to enforce an exact budget.

The registered primary plan is in `docs/joint_protocol.md`. Supporting records in `results/joint/` include 51 rows in `fits.csv`, 27 rows in `interventions.csv`, cell means, factorial contrasts, diagnostics, and the source hash in `environment.json`. The supplemental static-control and usage protocols live in this directory. Their hash receipts precede their respective measurements.

## Joint optimization

The fixed controls always retain the latest four symbols, form adjacent pairs, execute two steps, and choose the expert by query index. Learned memory chooses a four-symbol bank. Learned grouping chooses a pairing of retained symbols. All backbones also receive the header. The action space supplies the task-relevant possible layouts and bank sizes. The model learns which option to use.

| Policy | Held-out accuracy, mean % | Compact linear MACs per input |
|---|---:|---:|
| Fixed factorial baseline | 35.51 | 1,440 |
| D | 36.21 | 1,520 |
| G | 56.64 | 1,520 |
| R | 41.68 | 1,520 |
| M | 59.06 | 1,520 |
| DG | 57.82 | 1,536 |
| DR | 42.07 | 1,536 |
| DM | 57.39 | 1,536 |
| GM | 100.00 | 1,536 |
| Full DGRM | 91.17 | 1,568 |
| Informed policy with trained backbone | 99.84 | 1,456 |

All 16 factorial cells executed two recurrent steps per input on average, with the same 1,568 linear multiply-accumulates when all four controller outputs were physically evaluated. Compact counts include only controller columns needed in deployment. Allocation, copying, sorting and nonlinearities fall outside those counts. The hand-designed control decodes the four latent factors from the header using its known linear basis. The full training implementation enumerates all branches and computes expected prediction loss. The hard inference implementation executes only the selected expert for the chosen number of passes.

Full accuracy minus the four singleton accuracies plus three times baseline was +4.09 percentage points, with a paired 95% t interval from −5.90 to +14.09. The exploratory grouping-by-memory interaction averaged across the other switches was +15.97 points [11.10, 20.85]. The six interaction intervals are unadjusted. Three independent training seeds shared one test set. The grouping-plus-memory configuration dominated the full system on both mean accuracy and compact MACs. Absolute accuracy risks can saturate near 100%, and interactions on the accuracy scale depend on the chosen metric.

## Controller interventions

Permuting each example's complete action tuple within its cohort preserved the histogram of routes and reduced full accuracy by 50.75 points [32.93, 68.57]. Permuting only the grouping head cost 32.74 [23.42, 42.06]. Permuting only the memory head cost 32.35 [23.94, 40.76]. Both recovered the planted noisy-header factors on all held-out cases. The depth-only penalty was 4.13 points [−3.34, 11.61]. Two seeds collapsed onto one expert. These experiments support the utility of the tested bank and pairing decisions. They leave input-dependent depth and expert specialization unresolved in this benchmark.

## Supplemental controls and scoped use

The fixed factorial baseline discards the first half of every stream. A later control supplied a conventional dense MLP with the complete input at comparable parameter and inference counts. The 1,607-parameter MLP used 1,556 linear MACs per example. On 8,192 separately drawn cases it averaged 25.09% after 240 updates and 30.18% after 2,626 updates. The longer run used the same 15,360 unique training examples and nearly matched the factorial model's nominal training MACs. GM scored 100.00% and full scored 91.32% there. This comparison uses one fixed MLP architecture and one learning rate. The informed control demonstrates how effectively a deliberately structured human design can use the same task prior.

The online GM checkpoint reproduced its original accuracy and cross-entropy. After checkpoint selection, a further 10,000 independent draws from the same generator yielded 9,999 correct single-example predictions. The Wilson 95% interval on this fresh set is 99.943%–99.998%. This supports using the artifact as a classifier for this four-class encoding. It provides no evidence for natural-language use, transfer to another task, or general reasoning.
