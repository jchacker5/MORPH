# Boolean program execution

The four reasoning studies implement experiments 1, 6, 7 and 8. Input consists of an initial bit, a program scattered among distractor tokens, and a final query. The four program instructions are identity, inversion, set-zero and set-one. All Transformer weights are learned from initialization. A decoder readout on the last position predicts the output bit. The final query can attend to all preceding positions.

| Study | Launcher | Configuration | Primary use |
|---|---|---|---|
| Original small pilot and short scale ladder | `reasoning/suite.py all` | 21 pilot models and 18 scaled models | Prespecified initial comparison |
| Longer small-model replication | `reasoning/optimize.py` | 21 models, 2,400 updates each | Compare optimized baselines, controller interventions, and distribution shifts |
| Longer scaling replication | `reasoning/extended.py` | 45 models, 160 updates each | Compare fixed-depth 1–4 with adaptive depth at approximately 10M, 30M and 100M parameters |
| Engineering qualification | `reasoning/qualify.py` | Two seed-17 checkpoints, 8,000 updates each | Independently test the retained models on the specified task |

Run each using the project Python environment, for example `.venv/bin/python reasoning/suite.py all`. Each launcher resumes at the first absent completed JSON file. A new output directory/project copy is needed to repeat all training from scratch. Decisions, evaluation streams, and stopping rules are in `docs/reasoning_protocol.md` and the three extension protocols in this directory. Their local freeze receipts are under `results/reasoning`.

## Evaluation

`run.py` writes one JSON report and compressed per-example CSV per fitted model. Dates and hashes for each registered protocol accompany the results. The same test drawings are shared by every model within a stage. Fresh independent streams are used between stages. The adaptive model is trained using soft probabilities over exits. In deployed inference its `compact` path removes stopped examples before executing the next Transformer pass. The forced-depth and shuffled-allocation interventions separately evaluate all four exits for measurement. A seed-wise fixed-policy envelope takes the best measured accuracy obtainable using one fixed model or a mixture randomized independently of the input. This attained envelope can favor fixed models because it is computed from held-out results rather than a separate selection set.

`summarize.py` aggregates the original and two longer comparisons. `report.py` generates the engineering qualification statistics, manuscript supplement and figure. The `verify.py --full` check re-derives compute for each row, recreates final labels and counts from the recorded seeds, reconstructs summary counts and checks the saved weights against stored predictions. The `check_costs.py --repair` mode is a documented migration of the initial 32-bit integer cost field using the original recorded executed depths. It does not change any prediction, depth, training trajectory, or checkpoint.

## Inference

From the project root run:

```sh
.venv/bin/python reasoning/predict.py 1 not zero not --checkpoint adaptive --length 64
.venv/bin/python reasoning/predict.py 1 not zero not --checkpoint fixed2 --length 64
```

The output includes the predicted bit, exact interpreter answer and actual pass count. The optional seed changes placement and distractors. Checkpoint selectors `pilot` and `optimized` select the earlier 120- and 2,400-update adaptive models. Consult `results/reasoning/findings.md` for distribution-specific evidence before relying on any prediction.
