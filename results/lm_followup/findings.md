# Small language model follow-up

This follow-up used a configuration frozen after the earlier time-limited search. It held update counts constant for the replay and strengthened the paired comparison with 4,096 updates per model. All four recurrent architectures saw the same training windows within each seed. The frozen three-seed primary outcome was bits per byte on the 154 sonnets. The external test contained 87,744 scored bytes, with no exact 65-byte test window in the training corpus. The source, cleaned data hash and extraction code accompany the protocol. [Project Gutenberg edition](https://www.gutenberg.org/ebooks/1041) · [Pinned source](https://github.com/GITenberg/Shakespeare-s-Sonnets_1041/blob/b6da8a3e73e7081cf71c57e930001d0b92a8f644/1041.txt)

## Matched-update validation replay

| Trial | Original 30 s steps | Original penalized BPB | Replay steps | Replay BPB | Replay mean passes | Replay penalized BPB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5228 | 2.5537 | 2048 | 2.8003 | 1.000 | 2.8003 |
| 2 | 3947 | 2.8791 | 2048 | 2.9994 | 4.000 | 3.1594 |
| 3 | 3070 | 2.6887 | 2048 | 2.7769 | 1.000 | 2.7769 |
| 4 | 1870 | 2.7648 | 2048 | 2.7375 | 1.000 | 2.7375 |

The original protocol selected trial 1. Trial 4 scored lowest in the equal-update replay. All longer runs used the original selection.

## Three-seed paired experiment

| Pass policy | Parameters | External BPB | Repeat test BPB | External mean passes | Train TFLOPs | External inference MFLOPs/window | Batch-one ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 117,696 | 2.8458 | 3.0736 | 1.000 | 3.092 | 15.729 | 6.528 |
| 2 | 117,696 | 2.6247 | 2.8683 | 2.000 | 5.721 | 29.098 | 8.003 |
| 4 | 117,696 | 2.5879 | 2.8696 | 4.000 | 10.978 | 55.837 | 7.235 |
| adaptive | 117,793 | 2.5975 | 2.8750 | 1.000 | 12.370 | 15.729 | 8.423 |

Adaptive minus fixed two-pass external BPB averaged -0.0271, with a 95% paired Student interval [-0.1534, +0.0992]. Per-seed differences were -0.0821, -0.0176, +0.0183. Intervals have two degrees of freedom. The neural runs matched updates and input bytes. Four adaptive training exits incur extra FLOPs.

Count models trained on exactly 892,315 Tiny Shakespeare bytes gave 3.4801 BPB (bigram) and 2.4299 BPB (three-byte history) on the sonnets. Their repeat-test scores were 3.6251 and 2.9084. Each reported latency averages three seeds' median batch-one times across 40 synchronized windows with shared machine load. FLOP estimates count dense matrix arithmetic and exclude scalar kernels, data movement and optimizer updates.

## Five-minute capstone

Seed 7 completed 7,567 optimizer steps and consumed 7,748,608 input bytes in 300.022 seconds. Validation BPB was 2.4864, external BPB 2.4910, and repeated test BPB 2.7894. External mean depth was 1.000. This single shared-hardware duration run uses the earlier development seed.

## Frozen inference examples

Both retained models reproduced all eight checked logits/depths on reload and identical bytes on two generation calls with the registered prompt, seed and temperature. Each sample contains the original prompt followed by 160 sampled bytes. The raw hex bytes are in the per-model JSON.

4,096-step seed 11

~~~~text
To be, or not to be;
And I knee the thou tell ack, bind lords not in the ap his in the to ginnels from of thy his the hath hone
Ind weo.
KING RICHARD IIII:
When this aulls
That fr
~~~~

300-second seed 7

~~~~text
To be, or not to be;
And I knersh well be thine will Aufide-shepher?

ELBOHNOS:
I do my but offices me off the both,
Un come, take have inking, resom gis'd when
thiness, which muc
~~~~

## Reproduction

Run from the project root with the study environment. The corpus is included and its digest is checked before use.

~~~sh
.venv/bin/python -m lm_followup.run counts
.venv/bin/python -m lm_followup.run replay
.venv/bin/python -m lm_followup.run long
.venv/bin/python -m lm_followup.run capstone
.venv/bin/python -m lm_followup.analyze
.venv/bin/python lm/generate.py --small --weights results/lm_followup/long-adaptive-11.safetensors --prompt "To be, or not to be" --seed 73 --temperature .8 --bytes 160
~~~

The five-minute run depends on system throughput. Existing completed cells are reused. Use a fresh output directory in an isolated project copy for an independent replication and retain the protocol and digest. The sonnet scores measure related-author literary next-byte prediction. They do not evaluate instruction following, factuality, or useful answers to questions.
