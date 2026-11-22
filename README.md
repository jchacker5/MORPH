# MORPH
## Learning to Organize Computation in Neural Networks

**Joseph Defendre · Research report · November 21, 2026**

MORPH tests whether learning how much computation to spend, which inputs to group, which expert to use, and which records to retain improves predictive quality at a matched cost. This independent project includes eight experimental questions, locally trained weights, protocols and seed-level results.

The grouping-and-memory classifier scored **9,999/10,000** on fresh examples from its synthetic task. On Boolean reasoning, the learned-depth Transformer scored **85.53%** and the random-depth baseline scored **95.66%** at training-range lengths in the 2,400-update follow-up, averaged over three seeds. Both scored 50.46% in the initial 120-update pilot. The study found a narrow use for learned organization and a strong reason to keep testing simple controls.

- [Read the paper](paper/MORPH.pdf), [view the manuscript](paper/MORPH.tex) or [read the model card](MODEL_CARD.md).
- [Download the replication package](https://github.com/jchacker5/MORPH/releases/download/v0.1.0/MORPH-replication.zip) for the source, inputs, results and all 12 saved checkpoints. [Release and individual downloads](https://github.com/jchacker5/MORPH/releases/tag/v0.1.0) · [SHA-256 checksums](https://github.com/jchacker5/MORPH/releases/download/v0.1.0/SHA256SUMS)
- [Read the original research agenda](docs/proposal.md), [inspect the protocols](docs/) and [per-seed results](results/). The paper identifies experiments added after the original runs.

## Try a saved model

The small grouping-and-memory classifier runs on a CPU. From the project root, run:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'numpy==2.5.5'
.venv/bin/python joint/infer.py --header 2 0 0 0 --tokens 0 1 2 3 2 0 3 1 --query 1
```

The predicted class is `1`. On Apple silicon, install the MLX dependencies and try the Boolean interpreter and the five-minute byte language model. Both weights ship in the Git checkout.

```sh
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python reasoning/predict.py 1 not zero not --checkpoint adaptive --length 64
.venv/bin/python lm/generate.py --small --weights results/lm_followup/capstone.safetensors --prompt "KING HENRY:" --bytes 160 --seed 73
```

## Contribute

Replications, counterexamples and failed experiments are welcome. Start with an independent replay of the CPU classifier, or test whether the learned grouping survives noisier inputs. [Contributor guide](CONTRIBUTING.md) · [Open research issues](https://github.com/jchacker5/MORPH/issues)

Joseph Defendre licenses the original code and weights under MIT and the paper and original results under CC BY 4.0. Separate terms apply to the bundled corpora and fonts. [License details](LICENSING.md) · [Third-party notices](THIRD_PARTY.md) · [Citation](CITATION.cff)

## Reproduce the study

Training and scoring use Python 3.12.12 and locked dependencies. The Transformer experiments ran on Apple MLX on an M4 with 32 GiB of unified memory. The mechanism and factorial probes used CPU execution. From the project root, stage a fresh research directory to preserve the archived results. This copies code, local data and original protocol/selection records. It retains the original discovery records and omits final fit outputs and weights so the launchers can train from scratch. Use the new directory as the working directory for the following commands, with the original virtual environment or a new one built from the same requirements.

```sh
.venv/bin/python reproduce.py ../MORPH-replication
```

| Experiment | Training and analysis entry points | Principal evidence |
|---|---|---|
| 1, 6, 7, 8 · Adaptive depth, allocation, extrapolation, scaling | [Reasoning guide](reasoning/README.md) | Per-model JSON, per-example compressed CSVs, fixed-compute frontiers, forced exits and shifts in [reasoning results](results/reasoning/) |
| 2, 3, 4 · Grouping, experts, retention | [Probe methods and commands](results/mechanisms/findings.md) | [Mechanism results](results/mechanisms/) with task labels, causal checks, byte counts and interventions |
| 5, 6 · Joint optimization and interventions | [Joint study guide](joint/README.md) | [16-cell means](results/joint/cell_means.csv), [paired contrasts](results/joint/contrasts_pp.csv), [fresh usability test](results/joint/usability.json) |
| Natural-text depth comparison | `python lm/finalize.py small`, `counts`, then `prototype` | [Short and 30M training](results/lm/), [validation trial history](results/lm/search/results.tsv) |
| Longer small LM, external text, 300-second capstone | `python -m lm_followup.run counts`, `replay`, `long`, `capstone`, then `python -m lm_followup.analyze` | [Independent follow-up](results/lm_followup/) |
| Independently trained shallow 30M controls | [Methods and commands](prototype_controls/README.md) | [30M comparator results](results/prototype_controls/) |

After the original LM fits and prototypes finish, run `python lm/summarize.py` and `python lm/verify.py` to regenerate seed intervals and verify the saved model scores, future-byte invariance, mixed-exit dispatch and generation. Use `python paper/plot.py`, `python paper/make_reasoning_tables.py`, `python paper/make_language_tables.py`, `python paper/make_profile_table.py` and `tectonic paper/MORPH.tex` from the archived project to rebuild the figures and PDF. Costs marked FLOPs or MACs are analytical estimates. Measured latency can vary substantially with other processes on the same Mac.

### Autoresearch

The development loop follows the validation metric, fixed training-time and keep/discard procedure in [Karpathy's pinned autoresearch program](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md). The upstream CUDA nanochat trainer was replaced by the MORPH MLX decoder, corpus and frozen scorer. The four discovery fits used 30 seconds each. A separate capstone used the original five-minute duration. Current confirmation commands intentionally reuse the selected configuration. To reproduce discovery itself, begin with a fresh search output folder in a disposable copy, then run `lm/trial.py` using the four archived `.plan.json` hypotheses and configuration edits in order. Wall-time trials reproduce a procedure, and need not finish the same number of updates.

The language training corpus is pinned by commit and SHA-256. The external Sonnets test is drawn from a pinned edition of Project Gutenberg ebook 1041. See the [third-party source notices](THIRD_PARTY.md) for both editions.
