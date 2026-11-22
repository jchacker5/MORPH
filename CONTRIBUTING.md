# Contributing to MORPH

Joseph Defendre leads MORPH. Independent replications, bug fixes and experiments that challenge a finding are welcome. You can contribute by checking one prediction or one table. The [paper](paper/MORPH.pdf) and [model card](MODEL_CARD.md) describe what each checkpoint was trained to do.

## Run a smoke check

From the repository root, create a Python 3.12 environment and try the small grouping-and-memory classifier. The weights are included in the checkout. This example needs only NumPy.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'numpy==2.5.5'
.venv/bin/python joint/infer.py --header 2 0 0 0 --tokens 0 1 2 3 2 0 3 1 --query 1
```

Expect class `1`, retained bank `1`, pair layout `1`, expert `1` and two recurrent steps. On an Apple-silicon Mac, the following check also verifies that executing only the selected paths produces the same logits as evaluating every path and selecting afterward.

```sh
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python joint/run.py --verify-only
```

View the [open contribution issues](https://github.com/jchacker5/MORPH/issues) to share a result or coordinate a replication.

## Good first contributions

| Question | A concrete contribution |
|---|---|
| Can another machine reproduce the small classifier? | Add a NumPy-only replay of the published 10,000-example draw. Use the existing checkpoint, publish the correct count and compare the first 128 predictions with the batched reference. The original run scored 9,999/10,000. [Usage protocol](joint/usage_check.md) |
| When does learning to group and retain records fail? | Vary the noise on the synthetic header while keeping examples paired. Compare the bank and pair choices of the grouping-and-memory checkpoint with the choices decoded by the known header transform. Report disagreement rates and classification accuracy at each noise level. [Data and controller design](docs/joint_protocol.md) |
| Does the fixed-depth comparison hold with independent model selection? | Select the fixed-depth policy on a validation split, freeze it, then compare it with adaptive depth on fresh Boolean programs at matched estimated cost. The published analysis also reports an envelope selected on held-out outcomes. [Reasoning evaluation](reasoning/README.md) |

## Share a result

A GitHub issue is enough for a replication or a bug report. Include the commit, command, hardware, software versions and what you observed. For a pull request, describe the change and the smallest check that exercises it.

For a new experiment, write down the question, baseline, metric, compute budget and stopping rule before reading test scores. Fit or select settings on training and validation data. Record the seeds, test denominator, results for each seed and uncertainty for paired comparisons. Distinguish estimated FLOPs from measured latency and state which budget the models share. A collapsed controller, slower execution or a baseline win is worth reporting.

Use a fresh directory for training so another researcher can still check the published result. Keep follow-up protocols and outputs alongside a link to the original comparison. The [reproduction guide](README.md) gives the staging command and entry points for each experiment.

Contributed code and weights use MIT. Original paper edits and original experimental results use CC BY 4.0. Please identify the source and terms of any third-party material in a proposed change. [License details](LICENSING.md)

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md). Use [Discussions](https://github.com/jchacker5/MORPH/discussions) for questions or to propose an experiment before running it.
