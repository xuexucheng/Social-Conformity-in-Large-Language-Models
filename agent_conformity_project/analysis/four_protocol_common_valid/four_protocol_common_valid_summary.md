# Four-protocol common-valid analysis

All four protocol metrics within a model-dataset cell use the exact same
four-way common-valid subset S.

## qwen_csqa

- Dataset: CommonsenseQA
- Model: Qwen/Qwen2.5-3B-Instruct
- Common attempted N: 500
- Four-protocol common-valid N: 500
- Shared initial accuracy: 73.60%

| Protocol | N | Initial Acc | Final Acc | CR | HCR | Change | Target adoption | Beneficial revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 500 | 73.60% | 56.60% | 24.36% | 22.83% | 24.20% | 29.20% | 2.27% |
| exp2 | 500 | 73.60% | 32.60% | 58.76% | 55.71% | 56.60% | 61.40% | 2.27% |
| exp3 | 500 | 73.60% | 71.20% | 2.99% | 2.99% | 3.40% | 9.20% | 0.76% |
| exp4 | 500 | 73.60% | 62.40% | 17.52% | 15.76% | 16.80% | 22.80% | 1.52% |

- Exp4 malformed Step-5 rows: 0
- Exp4 Step5/final mismatches: 0

## qwen_mmlu

- Dataset: MMLU
- Model: Qwen/Qwen2.5-3B-Instruct
- Common attempted N: 500
- Four-protocol common-valid N: 500
- Shared initial accuracy: 63.20%

| Protocol | N | Initial Acc | Final Acc | CR | HCR | Change | Target adoption | Beneficial revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 500 | 63.20% | 40.20% | 42.26% | 36.08% | 38.60% | 49.40% | 1.63% |
| exp2 | 500 | 63.20% | 30.40% | 55.20% | 52.22% | 48.40% | 60.80% | 0.54% |
| exp3 | 500 | 63.20% | 60.40% | 4.85% | 4.43% | 4.20% | 17.60% | 0.00% |
| exp4 | 500 | 63.20% | 42.80% | 34.64% | 29.75% | 33.40% | 42.80% | 1.63% |

- Exp4 malformed Step-5 rows: 0
- Exp4 Step5/final mismatches: 0

## phi_csqa

- Dataset: CommonsenseQA
- Model: microsoft/Phi-3.5-mini-instruct
- Common attempted N: 500
- Four-protocol common-valid N: 497
- Shared initial accuracy: 74.65%

| Protocol | N | Initial Acc | Final Acc | CR | HCR | Change | Target adoption | Beneficial revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 497 | 74.65% | 23.94% | 67.30% | 66.85% | 64.99% | 69.01% | 0.79% |
| exp2 | 497 | 74.65% | 12.47% | 81.74% | 82.48% | 79.07% | 82.70% | 1.59% |
| exp3 | 497 | 74.65% | 58.75% | 15.29% | 14.29% | 22.54% | 19.52% | 3.17% |
| exp4 | 497 | 74.65% | 30.78% | 50.74% | 49.60% | 60.16% | 52.52% | 6.35% |

- Exp4 malformed Step-5 rows: 0
- Exp4 Step5/final mismatches: 0

## phi_mmlu

- Dataset: MMLU
- Model: microsoft/Phi-3.5-mini-instruct
- Common attempted N: 500
- Four-protocol common-valid N: 478
- Shared initial accuracy: 66.95%

| Protocol | N | Initial Acc | Final Acc | CR | HCR | Change | Target adoption | Beneficial revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 478 | 66.95% | 41.63% | 36.58% | 37.19% | 34.10% | 43.93% | 1.27% |
| exp2 | 478 | 66.95% | 19.67% | 67.22% | 68.44% | 61.72% | 70.71% | 0.00% |
| exp3 | 478 | 66.95% | 43.72% | 28.50% | 29.69% | 32.01% | 35.36% | 2.53% |
| exp4 | 478 | 66.95% | 41.84% | 33.25% | 33.44% | 37.24% | 39.96% | 5.06% |

- Exp4 malformed Step-5 rows: 0
- Exp4 Step5/final mismatches: 0

## gemma_csqa

- Dataset: CommonsenseQA
- Model: google/gemma-2-2b-it
- Common attempted N: 500
- Four-protocol common-valid N: 500
- Shared initial accuracy: 68.80%

| Protocol | N | Initial Acc | Final Acc | CR | HCR | Change | Target adoption | Beneficial revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 500 | 68.80% | 67.60% | 2.35% | 1.45% | 2.80% | 8.60% | 0.64% |
| exp2 | 500 | 68.80% | 66.40% | 4.49% | 3.49% | 4.60% | 10.60% | 0.64% |
| exp3 | 500 | 68.80% | 61.60% | 11.11% | 9.59% | 13.60% | 16.80% | 3.21% |
| exp4 | 500 | 68.80% | 65.40% | 5.56% | 4.36% | 17.80% | 9.20% | 14.10% |

- Exp4 malformed Step-5 rows: 0
- Exp4 Step5/final mismatches: 0

## gemma_mmlu

- Dataset: MMLU
- Model: google/gemma-2-2b-it
- Common attempted N: 500
- Four-protocol common-valid N: 499
- Shared initial accuracy: 55.71%

| Protocol | N | Initial Acc | Final Acc | CR | HCR | Change | Target adoption | Beneficial revision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 499 | 55.71% | 55.11% | 1.64% | 1.80% | 2.20% | 15.63% | 1.36% |
| exp2 | 499 | 55.71% | 54.51% | 2.80% | 2.16% | 2.40% | 16.63% | 0.00% |
| exp3 | 499 | 55.71% | 48.10% | 12.15% | 12.23% | 12.83% | 24.45% | 0.90% |
| exp4 | 499 | 55.71% | 53.11% | 4.21% | 5.40% | 9.82% | 16.43% | 6.33% |

- Exp4 malformed Step-5 rows: 0
- Exp4 Step5/final mismatches: 0
