# P3 Forced-option distractor plausibility stratification

Terciles are defined within each model×dataset cell on the four-protocol common-valid item set using the pre-social target-option probability.

Low: p <= q33; Medium: q33 < p <= q67; High: p > q67.

Paired bootstrap: 10,000 resamples, base seed 12345.

## Qwen2.5-3B — CommonsenseQA

Common-valid N = 500; q33 = 1.1238468e-07; q67 = 1.7603416e-06.

| Stratum | N | Exp1 CR/HCR | Exp2 CR/HCR | Exp3 CR/HCR | Exp4 CR/HCR |
|---|---:|---:|---:|---:|---:|
| Low | 167 | 15.57/16.45 | 49.10/50.00 | 1.20/1.32 | 9.58/9.87 |
| Medium | 166 | 21.08/20.30 | 51.20/48.87 | 1.81/1.50 | 12.65/12.03 |
| High | 167 | 39.26/38.55 | 80.00/77.11 | 6.67/8.43 | 33.33/32.53 |

CR/HCR values above are percentages.

| Stratum | Contrast | N(CR) | ΔCR (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|
| Low | exp1→exp2 | 167 | 33.53 | [25.75, 41.32] |
| Low | exp2→exp3 | 167 | -47.90 | [-55.09, -40.72] |
| Medium | exp1→exp2 | 166 | 30.12 | [21.08, 38.55] |
| Medium | exp2→exp3 | 166 | -49.40 | [-57.23, -41.57] |
| High | exp1→exp2 | 135 | 40.74 | [31.11, 50.37] |
| High | exp2→exp3 | 135 | -73.33 | [-80.74, -65.93] |

## Qwen2.5-3B — MMLU

Common-valid N = 500; q33 = 8.315266e-07; q67 = 0.00025917173.

| Stratum | N | Exp1 CR/HCR | Exp2 CR/HCR | Exp3 CR/HCR | Exp4 CR/HCR |
|---|---:|---:|---:|---:|---:|
| Low | 167 | 22.75/21.92 | 38.32/36.99 | 1.80/2.05 | 18.56/15.07 |
| Medium | 166 | 48.80/43.36 | 59.64/61.95 | 5.42/5.31 | 35.54/34.51 |
| High | 167 | 64.00/57.89 | 76.00/71.93 | 9.00/8.77 | 60.00/57.89 |

CR/HCR values above are percentages.

| Stratum | Contrast | N(CR) | ΔCR (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|
| Low | exp1→exp2 | 167 | 15.57 | [7.78, 23.95] |
| Low | exp2→exp3 | 167 | -36.53 | [-43.71, -29.34] |
| Medium | exp1→exp2 | 166 | 10.84 | [1.20, 20.48] |
| Medium | exp2→exp3 | 166 | -54.22 | [-62.05, -46.39] |
| High | exp1→exp2 | 100 | 12.00 | [2.00, 22.00] |
| High | exp2→exp3 | 100 | -67.00 | [-76.00, -58.00] |

## Phi-3.5-mini — CommonsenseQA

Common-valid N = 497; q33 = 8.2955699e-10; q67 = 6.4738531e-07.

| Stratum | N | Exp1 CR/HCR | Exp2 CR/HCR | Exp3 CR/HCR | Exp4 CR/HCR |
|---|---:|---:|---:|---:|---:|
| Low | 166 | 56.02/57.24 | 75.90/76.32 | 11.45/10.53 | 41.57/42.11 |
| Medium | 165 | 64.63/66.41 | 82.32/85.16 | 13.41/12.50 | 48.17/50.00 |
| High | 166 | 83.69/83.52 | 87.94/89.01 | 21.99/23.08 | 64.54/61.54 |

CR/HCR values above are percentages.

| Stratum | Contrast | N(CR) | ΔCR (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|
| Low | exp1→exp2 | 166 | 19.88 | [13.86, 26.51] |
| Low | exp2→exp3 | 166 | -64.46 | [-72.29, -56.63] |
| Medium | exp1→exp2 | 164 | 17.68 | [10.98, 25.00] |
| Medium | exp2→exp3 | 164 | -68.90 | [-76.83, -60.98] |
| High | exp1→exp2 | 141 | 4.26 | [-2.13, 10.64] |
| High | exp2→exp3 | 141 | -65.96 | [-74.47, -56.74] |

## Phi-3.5-mini — MMLU

Common-valid N = 478; q33 = 2.5110001e-08; q67 = 0.0013867561.

| Stratum | N | Exp1 CR/HCR | Exp2 CR/HCR | Exp3 CR/HCR | Exp4 CR/HCR |
|---|---:|---:|---:|---:|---:|
| Low | 160 | 31.25/32.21 | 61.88/63.09 | 21.88/22.82 | 27.50/28.19 |
| Medium | 159 | 37.97/40.71 | 66.46/69.03 | 32.91/37.17 | 34.18/37.17 |
| High | 159 | 42.72/43.10 | 76.70/81.03 | 32.04/32.76 | 40.78/39.66 |

CR/HCR values above are percentages.

| Stratum | Contrast | N(CR) | ΔCR (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|
| Low | exp1→exp2 | 160 | 30.63 | [22.50, 38.75] |
| Low | exp2→exp3 | 160 | -40.00 | [-49.38, -30.63] |
| Medium | exp1→exp2 | 158 | 28.48 | [20.89, 36.08] |
| Medium | exp2→exp3 | 158 | -33.54 | [-43.67, -23.42] |
| High | exp1→exp2 | 103 | 33.98 | [23.30, 44.66] |
| High | exp2→exp3 | 103 | -44.66 | [-56.31, -32.04] |

## Gemma-2-2B — CommonsenseQA

Common-valid N = 500; q33 = 4.0051445e-05; q67 = 0.00066553234.

| Stratum | N | Exp1 CR/HCR | Exp2 CR/HCR | Exp3 CR/HCR | Exp4 CR/HCR |
|---|---:|---:|---:|---:|---:|
| Low | 167 | 0.60/0.00 | 0.00/0.00 | 2.40/2.08 | 0.60/0.00 |
| Medium | 166 | 1.81/2.42 | 4.22/5.65 | 4.82/6.45 | 6.02/5.65 |
| High | 167 | 5.19/2.63 | 10.37/6.58 | 29.63/28.95 | 11.11/10.53 |

CR/HCR values above are percentages.

| Stratum | Contrast | N(CR) | ΔCR (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|
| Low | exp1→exp2 | 167 | -0.60 | [-1.80, 0.00] |
| Low | exp2→exp3 | 167 | 2.40 | [0.60, 4.79] |
| Medium | exp1→exp2 | 166 | 2.41 | [0.60, 4.82] |
| Medium | exp2→exp3 | 166 | 0.60 | [-3.01, 4.22] |
| High | exp1→exp2 | 135 | 5.19 | [0.00, 11.11] |
| High | exp2→exp3 | 135 | 19.26 | [11.11, 27.41] |

## Gemma-2-2B — MMLU

Common-valid N = 499; q33 = 0.00051512844; q67 = 0.020487976.

| Stratum | N | Exp1 CR/HCR | Exp2 CR/HCR | Exp3 CR/HCR | Exp4 CR/HCR |
|---|---:|---:|---:|---:|---:|
| Low | 167 | 1.20/1.42 | 1.20/0.71 | 9.58/8.51 | 2.99/3.55 |
| Medium | 166 | 1.81/3.06 | 3.61/4.08 | 11.45/13.27 | 5.42/7.14 |
| High | 166 | 2.11/0.00 | 4.21/2.56 | 17.89/23.08 | 4.21/7.69 |

CR/HCR values above are percentages.

| Stratum | Contrast | N(CR) | ΔCR (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|
| Low | exp1→exp2 | 167 | 0.00 | [-1.80, 1.80] |
| Low | exp2→exp3 | 167 | 8.38 | [4.19, 13.17] |
| Medium | exp1→exp2 | 166 | 1.81 | [-0.60, 4.82] |
| Medium | exp2→exp3 | 166 | 7.83 | [3.01, 13.25] |
| High | exp1→exp2 | 95 | 2.11 | [0.00, 5.26] |
| High | exp2→exp3 | 95 | 13.68 | [7.37, 21.05] |
