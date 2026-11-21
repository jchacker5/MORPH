# MORPH model card

**Project and research concept by Joseph Defendre.** These checkpoints were trained from initialization on an Apple M4. Each solves a different problem. The grouping, routing and memory research probes are separate from the byte-level language models.

## Which model can I actually use?

| Artifact | Supported use | Held-out evidence |
|---|---|---|
| [Grouping + memory classifier](results/joint/gm_seed41.npz) | Predict a class 0–3 from the joint benchmark's four-dimensional header, eight symbols and query bit | 9,999/10,000 on a fresh draw after model selection. The task supplies the relevant decision cues. |
| [Fixed two-pass Boolean interpreter](results/reasoning/qualified-fixed2-reasoning-seed17.safetensors) | Classify bounded unary Boolean programs embedded in the study's synthetic format | 1,136/1,152 correct at training-range lengths. Accuracy fell to 65.63% at 1,024 tokens. |
| [Adaptive Boolean interpreter](results/reasoning/qualified-adaptive-reasoning-seed17.safetensors) | Inspect the returned prediction and actual number of recurrent passes | 1,087/1,152 correct at training-range lengths and 56.25% at 1,024. It missed the 95% accuracy qualification. |
| [Four-slot recall model](results/mechanisms/recall-seed11-k4.npz) | Retrieve a value for a queried key when exactly four prior records match a known cue | Reached 100% exact recall across the tested cue and position shifts. A 12-byte deterministic cue cache matched it. |
| [Five-minute byte LM](results/lm_followup/capstone.safetensors) | Predict bytes or generate short Shakespeare-like continuations locally | 2.789 bits/byte on plays and 2.491 on sonnets. All evaluated windows used one pass. The three-byte-history predictor scored 2.908 and 2.430 respectively. |
| [30M byte LM](results/lm/prototype/adaptive-11-2048.safetensors) | Run the larger experimental recurrent decoder; inspect the research-scale comparison | See the 30M results below. |

The narrow classifier and next-byte results demonstrate that useful models can be trained on this laptop. No checkpoint was instruction-tuned or tested for conversation, factual answering, or general-purpose assistance.

## Examples

Use the project Python environment from the project root. The first command needs NumPy only. The other examples use Apple MLX.

```sh
.venv/bin/python joint/infer.py --header 2 0 0 0 --tokens 0 1 2 3 2 0 3 1 --query 1
.venv/bin/python reasoning/predict.py 1 not zero not --checkpoint fixed2 --length 64
.venv/bin/python lm/generate.py --small --weights results/lm_followup/capstone.safetensors --prompt "To be, or not to be" --bytes 160 --temperature 0.8 --seed 73
.venv/bin/python lm/generate.py --prompt "KING HENRY:" --bytes 160 --temperature 0.8 --seed 73
```

The classifier example predicts class 1. The Boolean CLI includes the interpreter's ground truth alongside the model's prediction. The byte LM performs autoregressive sampling with a rolling 64-byte context. Every additional byte can require a fresh forward pass. The gate looks at up to 16 bytes at the start of that window. Short prompts use all available prompt bytes for the initial decision.

The first three lines of the five-minute model's fixed sample were:

```text
To be, or not to be;
And I knersh well be thine will Aufide-shepher?

ELBOHNOS:
```

It learns spelling fragments, layout, and speaker-like labels. The invented words and unreliable sentence meaning are visible even in a short sample. The independent external test used 87,744 scored bytes from 154 sonnets. The plays test used 111,504 scored bytes. Both are English literary works by the same author.

## Architecture and data

The scientific design distinguishes an adaptive training criterion from input-dependent execution. Recurrent models reuse the same block up to four times. A learned halting hazard is trained using the probability-weighted loss at each possible exit, plus a compute penalty. At inference the implementation stops and removes finished examples from the batch. A constant exit depth can still produce a functional predictor.

Tiny Shakespeare is pinned to char-rnn revision 6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e with contiguous 80/10/10 byte splits. The Sonnets edition is pinned to GITenberg revision b6da8a3e73e7081cf71c57e930001d0b92a8f644. Input bytes map to 256 output classes. The synthetic generators explicitly encode the cues and instruction vocabularies in each suite's documented format. There is no natural-language interface for the associative or Boolean classifiers.

