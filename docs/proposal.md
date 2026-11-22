# MORPH research proposal

**Joseph Defendre · Original proposal dated November 21, 2026**

MORPH stands for *Meta-Optimized Reconfigurable Processing Hierarchy*. The founding question is whether a neural network can learn to allocate and organize computation more efficiently than a human-designed fixed architecture. This public summary records the research agenda. The [dated experimental protocols](./) and [research report](../paper/MORPH.pdf) document the decisions and results for the first study.

## Hypothesis

Let a learned controller with parameters φ select a computational graph G(x; φ) for input x. The backbone with parameters θ makes its prediction using that graph. Train both the controller and the backbone on expected task loss plus a weighted cost for computation. An optional regularizer can penalize unstable or fragmented choices. A controller can draw on the intermediate representation and remaining resource budget when deciding whether to continue processing, which tokens to group, which expert to invoke or which records to retain.

The claim has an observable test. At a given inference budget, the learned policy should improve held-out predictive quality against strong fixed competitors. Permuting a learned decision among comparable inputs should hurt performance if its assignment matters. The advantage should survive independently varied sequence length and compositional difficulty. Training FLOPs, inference FLOPs, parameter counts and latency answer different questions and each requires its own comparison.

## Eight experiments

| Experiment | Question and planned comparison |
|---|---|
| 1. Adaptive depth | Compare one to four learned recurrent passes with fixed recurrent depths, random depth and a decoder with independent weights. Plot accuracy against inference FLOPs. |
| 2. Semantic grouping | Hide a known token hierarchy and compare learned groups with random, adjacent and fixed-stride assignments. Measure grouping recovery and conditional perplexity at equal compression. Enforce causality for sequential prediction. |
| 3. Expert routing | Mix arithmetic, syntax, copying and symbolic transformations. Compare learned, random, dense and conventional mixture-of-experts routing on quality and active parameters. |
| 4. Selective memory | Change which records will matter and where they occur. Compare learned retention, full attention, fixed windows and deterministic compression on recall, storage and latency. |
| 5. Joint optimization | Cross depth, grouping, routing and memory choices in a factorial design to estimate main effects and interactions at comparable cost. |
| 6. Emergent specialization | Relate controller decisions to independently defined difficulty. Force or permute decisions and measure the change in predictions at a controlled budget. |
| 7. Generalization | Train at lengths 32–128 and test at 256, 512 and 1,024. Vary compositional depth separately to distinguish these shifts. |
| 8. Scaling | Start with a laptop-scale pilot and compare the strongest configurations at a provisional 10M, 30M and 100M parameters. Use multiple seeds and report how training budgets change the comparison. |

## What would justify a claim?

The original proposal asked for three reproducible findings: an improved quality–compute frontier, an input-dependent allocation that causally helps, and generalization of that allocation beyond training. It called for training from scratch, recording hypotheses and budgets before examining outcomes, reporting uncertainty, and publishing unfavorable findings. The proposed MLX language prototype targeted 30M parameters and a maximum of four recurrent passes. The full framework was intended to grow in stages so that individual mechanisms could be tested separately.

The released report gives the actual parameter sizes, training budgets, scope of each task and outcomes, including where simpler policies won. [Paper](../paper/MORPH.pdf) · [Model card](../MODEL_CARD.md)
