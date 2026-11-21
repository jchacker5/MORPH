# Independent shallow-decoder results

Joseph Defendre

Six separately trained fixed-depth 30M models produced eight endpoints. All seeds shared the corresponding adaptive model’s initial backbone, training windows and optimizer schedule. The play test covered 111,504 scored bytes and the Sonnets test covered 87,744. All comparisons below are paired on the displayed seeds. Positive BPB differences favor the fixed decoder.

| Updates | Corpus | Fixed passes | Paired seeds | Adaptive BPB | Fixed BPB | Difference in BPB [95% interval] | Adaptive / fixed inference FLOPs | Adaptive / fixed training FLOPs |
|---:|---|---:|---|---:|---:|---|---:|---:|
| 512 | plays | 1 | 11, 23, 37 | 4.4339 | 4.2092 | +0.2247 [-0.1122, +0.5616] | 3.967× | 4.000× |
| 512 | plays | 2 | 11, 23, 37 | 4.4339 | 4.1974 | +0.2365 [-0.1855, +0.6585] | 1.995× | 2.011× |
| 512 | plays | 4 | 11, 23, 37 | 4.4339 | 4.6910 | -0.2571 [-0.7407, +0.2265] | 1.000× | 1.008× |
| 512 | sonnets | 1 | 11 | 4.3210 | 4.1197 | +0.2013 | 3.967× | 4.000× |
| 512 | sonnets | 2 | 11 | 4.3210 | 4.1277 | +0.1934 | 1.995× | 2.011× |
| 2048 | plays | 1 | 11 | 3.8675 | 3.7563 | +0.1112 | 3.967× | 4.000× |
| 2048 | plays | 2 | 11 | 3.8675 | 3.7401 | +0.1275 | 1.995× | 2.011× |
| 2048 | plays | 4 | 11 | 3.8675 | 3.8327 | +0.0349 | 1.000× | 1.008× |
| 2048 | sonnets | 1 | 11 | 3.7692 | 3.6407 | +0.1285 | 3.967× | 4.000× |
| 2048 | sonnets | 2 | 11 | 3.7692 | 3.6278 | +0.1414 | 1.995× | 2.011× |

The 512-update play comparisons have paired two-sided Student intervals over three training seeds. The external comparisons use the single available adaptive seed. All protocols chose checkpoints by step count before evaluating these controls. Model-level scores and exact histograms are in `all_scores.csv`. The SHA-256 and original-test comparisons for the two adaptive checkpoint reloads are recorded separately.
