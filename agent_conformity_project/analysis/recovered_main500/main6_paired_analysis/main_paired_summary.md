# Main protocol paired analysis

Bootstrap repetitions: 10000
Bootstrap seed: 12345
Multiplicity family: all prespecified model-dataset cells and protocol contrasts, corrected separately within each outcome

All deltas below are condition 2 minus condition 1.

- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / accuracy: N=500, 0.5660 -> 0.3260, delta=-24.00 pp, 95% CI [-28.40, -19.60], McNemar b=137, c=17, p=1.72872e-24, Holm p=1.38297e-23
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / target_adoption: N=500, 0.2920 -> 0.6140, delta=+32.20 pp, 95% CI [+27.60, +37.00], McNemar b=19, c=180, p=4.48412e-34, Holm p=4.03571e-33
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / change_rate: N=500, 0.2420 -> 0.5660, delta=+32.40 pp, 95% CI [+27.60, +37.00], McNemar b=19, c=181, p=2.47584e-34, Holm p=2.22826e-33
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / conformity_rate: N=468, 0.2436 -> 0.5876, delta=+34.40 pp, 95% CI [+29.49, +39.53], McNemar b=19, c=180, p=4.48412e-34, Holm p=4.03571e-33
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / harmful_conformity: N=368, 0.2283 -> 0.5571, delta=+32.88 pp, 95% CI [+27.45, +38.59], McNemar b=15, c=136, p=1.41708e-25, Holm p=1.27537e-24
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / beneficial_revision: N=132, 0.0227 -> 0.0227, delta=+0.00 pp, 95% CI [-2.27, +2.27], McNemar b=1, c=1, p=1, Holm p=1
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / accuracy: N=500, 0.3260 -> 0.7120, delta=+38.60 pp, 95% CI [+34.40, +43.00], McNemar b=3, c=196, p=3.26982e-54, Holm p=3.5968e-53
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / target_adoption: N=500, 0.6140 -> 0.0920, delta=-52.20 pp, 95% CI [-56.60, -47.80], McNemar b=261, c=0, p=5.39761e-79, Holm p=5.93737e-78
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / change_rate: N=500, 0.5660 -> 0.0340, delta=-53.20 pp, 95% CI [-57.60, -48.80], McNemar b=266, c=0, p=1.68675e-80, Holm p=2.0241e-79
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / conformity_rate: N=468, 0.5876 -> 0.0299, delta=-55.77 pp, 95% CI [-60.26, -51.28], McNemar b=261, c=0, p=5.39761e-79, Holm p=5.93737e-78
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / harmful_conformity: N=368, 0.5571 -> 0.0299, delta=-52.72 pp, 95% CI [-57.88, -47.55], McNemar b=194, c=0, p=7.96546e-59, Holm p=8.76201e-58
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / beneficial_revision: N=132, 0.0227 -> 0.0076, delta=-1.52 pp, 95% CI [-4.55, +1.52], McNemar b=3, c=1, p=0.625, Holm p=1
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / accuracy: N=500, 0.4020 -> 0.3040, delta=-9.80 pp, 95% CI [-13.80, -5.80], McNemar b=79, c=30, p=2.95585e-06, Holm p=1.18234e-05
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / target_adoption: N=500, 0.4940 -> 0.6080, delta=+11.40 pp, 95% CI [+6.60, +16.00], McNemar b=46, c=103, p=3.45107e-06, Holm p=1.38043e-05
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / change_rate: N=500, 0.3860 -> 0.4840, delta=+9.80 pp, 95% CI [+5.20, +14.60], McNemar b=51, c=100, p=8.21285e-05, Holm p=0.000246386
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / conformity_rate: N=433, 0.4226 -> 0.5520, delta=+12.93 pp, 95% CI [+7.39, +18.24], McNemar b=46, c=102, p=4.80527e-06, Holm p=1.92211e-05
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / harmful_conformity: N=316, 0.3608 -> 0.5222, delta=+16.14 pp, 95% CI [+10.13, +22.47], McNemar b=28, c=79, p=8.42001e-07, Holm p=3.36801e-06
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / beneficial_revision: N=184, 0.0163 -> 0.0054, delta=-1.09 pp, 95% CI [-2.72, +0.00], McNemar b=2, c=0, p=0.5, Holm p=1
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / accuracy: N=500, 0.3040 -> 0.6040, delta=+30.00 pp, 95% CI [+25.80, +34.20], McNemar b=3, c=153, p=1.38568e-41, Holm p=1.38568e-40
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / target_adoption: N=500, 0.6080 -> 0.1760, delta=-43.20 pp, 95% CI [-47.60, -38.80], McNemar b=220, c=4, p=7.71424e-60, Holm p=7.71424e-59
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / change_rate: N=500, 0.4840 -> 0.0420, delta=-44.20 pp, 95% CI [-48.60, -39.60], McNemar b=223, c=2, p=9.43103e-64, Holm p=9.43103e-63
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / conformity_rate: N=433, 0.5520 -> 0.0485, delta=-50.35 pp, 95% CI [-54.97, -45.50], McNemar b=220, c=2, p=7.34542e-63, Holm p=7.34542e-62
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / harmful_conformity: N=316, 0.5222 -> 0.0443, delta=-47.78 pp, 95% CI [-53.48, -42.09], McNemar b=153, c=2, p=5.29472e-43, Holm p=5.29472e-42
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / beneficial_revision: N=184, 0.0054 -> 0.0000, delta=-0.54 pp, 95% CI [-1.63, +0.00], McNemar b=1, c=0, p=1, Holm p=1
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / accuracy: N=497, 0.2394 -> 0.1247, delta=-11.47 pp, 95% CI [-14.69, -8.45], McNemar b=63, c=6, p=4.47353e-13, Holm p=2.68412e-12
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / target_adoption: N=497, 0.6901 -> 0.8270, delta=+13.68 pp, 95% CI [+10.26, +17.30], McNemar b=12, c=80, p=1.7156e-13, Holm p=1.02936e-12
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / change_rate: N=497, 0.6499 -> 0.7907, delta=+14.08 pp, 95% CI [+10.66, +17.51], McNemar b=7, c=77, p=5.13756e-16, Holm p=3.08254e-15
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / conformity_rate: N=471, 0.6730 -> 0.8174, delta=+14.44 pp, 95% CI [+10.62, +18.26], McNemar b=12, c=80, p=1.7156e-13, Holm p=1.02936e-12
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / harmful_conformity: N=371, 0.6685 -> 0.8248, delta=+15.63 pp, 95% CI [+11.59, +19.95], McNemar b=6, c=64, p=2.44273e-13, Holm p=1.46564e-12
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / beneficial_revision: N=126, 0.0079 -> 0.0159, delta=+0.79 pp, 95% CI [-1.59, +3.17], McNemar b=1, c=2, p=1, Holm p=1
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / accuracy: N=497, 0.1247 -> 0.5875, delta=+46.28 pp, 95% CI [+41.65, +51.11], McNemar b=8, c=238, p=5.42486e-60, Holm p=6.50984e-59
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / target_adoption: N=497, 0.8270 -> 0.1952, delta=-63.18 pp, 95% CI [-67.81, -58.55], McNemar b=324, c=10, p=2.45e-82, Holm p=2.94e-81
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / change_rate: N=497, 0.7907 -> 0.2254, delta=-56.54 pp, 95% CI [-61.17, -51.91], McNemar b=293, c=12, p=3.47529e-71, Holm p=3.82282e-70
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / conformity_rate: N=471, 0.8174 -> 0.1529, delta=-66.45 pp, 95% CI [-71.13, -61.78], McNemar b=323, c=10, p=4.75375e-82, Holm p=5.7045e-81
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / harmful_conformity: N=371, 0.8248 -> 0.1429, delta=-68.19 pp, 95% CI [-73.32, -63.07], McNemar b=259, c=6, p=1.5688e-68, Holm p=1.88256e-67
- CommonsenseQA / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / beneficial_revision: N=126, 0.0159 -> 0.0317, delta=+1.59 pp, 95% CI [-2.38, +5.56], McNemar b=2, c=4, p=0.6875, Holm p=1
- MMLU / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / accuracy: N=494, 0.4211 -> 0.2045, delta=-21.66 pp, 95% CI [-25.51, -17.61], McNemar b=114, c=7, p=5.05491e-26, Holm p=4.54942e-25
- MMLU / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / target_adoption: N=494, 0.4393 -> 0.7024, delta=+26.32 pp, 95% CI [+21.86, +30.77], McNemar b=15, c=145, p=6.82617e-28, Holm p=4.77832e-27
- MMLU / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / change_rate: N=494, 0.3401 -> 0.6113, delta=+27.13 pp, 95% CI [+22.67, +31.38], McNemar b=11, c=145, p=5.50348e-31, Holm p=4.40279e-30
- MMLU / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / conformity_rate: N=435, 0.3655 -> 0.6667, delta=+30.11 pp, 95% CI [+25.29, +34.94], McNemar b=13, c=144, p=4.07809e-29, Holm p=3.26247e-28
- MMLU / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / harmful_conformity: N=333, 0.3694 -> 0.6757, delta=+30.63 pp, 95% CI [+24.92, +36.34], McNemar b=9, c=111, p=1.70929e-23, Holm p=1.36743e-22
- MMLU / microsoft/Phi-3.5-mini-instruct / exp1 -> exp2 / beneficial_revision: N=161, 0.0124 -> 0.0000, delta=-1.24 pp, 95% CI [-3.11, +0.00], McNemar b=2, c=0, p=0.5, Holm p=1
- MMLU / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / accuracy: N=489, 0.2025 -> 0.4356, delta=+23.31 pp, 95% CI [+18.20, +28.22], McNemar b=33, c=147, p=2.24311e-18, Holm p=1.57018e-17
- MMLU / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / target_adoption: N=489, 0.7035 -> 0.3579, delta=-34.56 pp, 95% CI [-39.88, -29.24], McNemar b=206, c=37, p=1.2868e-29, Holm p=1.02944e-28
- MMLU / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / change_rate: N=489, 0.6115 -> 0.3231, delta=-28.83 pp, 95% CI [-34.36, -23.31], McNemar b=188, c=47, p=3.67376e-21, Holm p=2.57163e-20
- MMLU / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / conformity_rate: N=430, 0.6674 -> 0.2884, delta=-37.91 pp, 95% CI [-43.72, -31.86], McNemar b=198, c=35, p=8.3062e-29, Holm p=5.81434e-28
- MMLU / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / harmful_conformity: N=328, 0.6768 -> 0.2988, delta=-37.80 pp, 95% CI [-44.82, -30.79], McNemar b=153, c=29, p=1.49635e-21, Holm p=1.04744e-20
- MMLU / microsoft/Phi-3.5-mini-instruct / exp2 -> exp3 / beneficial_revision: N=161, 0.0000 -> 0.0248, delta=+2.48 pp, 95% CI [+0.62, +4.97], McNemar b=0, c=4, p=0.125, Holm p=1
- CommonsenseQA / google/gemma-2-2b-it / exp1 -> exp2 / accuracy: N=500, 0.6760 -> 0.6640, delta=-1.20 pp, 95% CI [-2.40, +0.00], McNemar b=8, c=2, p=0.109375, Holm p=0.21875
- CommonsenseQA / google/gemma-2-2b-it / exp1 -> exp2 / target_adoption: N=500, 0.0860 -> 0.1060, delta=+2.00 pp, 95% CI [+0.40, +3.80], McNemar b=5, c=15, p=0.0413895, Holm p=0.0827789
- CommonsenseQA / google/gemma-2-2b-it / exp1 -> exp2 / change_rate: N=500, 0.0280 -> 0.0460, delta=+1.80 pp, 95% CI [+0.20, +3.60], McNemar b=5, c=14, p=0.0635681, Holm p=0.127136
- CommonsenseQA / google/gemma-2-2b-it / exp1 -> exp2 / conformity_rate: N=468, 0.0235 -> 0.0449, delta=+2.14 pp, 95% CI [+0.43, +4.06], McNemar b=5, c=15, p=0.0413895, Holm p=0.0827789
- CommonsenseQA / google/gemma-2-2b-it / exp1 -> exp2 / harmful_conformity: N=344, 0.0145 -> 0.0349, delta=+2.03 pp, 95% CI [+0.29, +4.07], McNemar b=2, c=9, p=0.0654297, Holm p=0.130859
- CommonsenseQA / google/gemma-2-2b-it / exp1 -> exp2 / beneficial_revision: N=156, 0.0064 -> 0.0064, delta=+0.00 pp, 95% CI [+0.00, +0.00], McNemar b=0, c=0, p=1, Holm p=1
- CommonsenseQA / google/gemma-2-2b-it / exp2 -> exp3 / accuracy: N=500, 0.6640 -> 0.6160, delta=-4.80 pp, 95% CI [-7.20, -2.40], McNemar b=32, c=8, p=0.000182166, Holm p=0.000546497
- CommonsenseQA / google/gemma-2-2b-it / exp2 -> exp3 / target_adoption: N=500, 0.1060 -> 0.1680, delta=+6.20 pp, 95% CI [+3.60, +8.80], McNemar b=8, c=39, p=5.53962e-06, Holm p=1.66189e-05
- CommonsenseQA / google/gemma-2-2b-it / exp2 -> exp3 / change_rate: N=500, 0.0460 -> 0.1360, delta=+9.00 pp, 95% CI [+6.20, +12.00], McNemar b=8, c=53, p=2.98644e-09, Holm p=1.19458e-08
- CommonsenseQA / google/gemma-2-2b-it / exp2 -> exp3 / conformity_rate: N=468, 0.0449 -> 0.1111, delta=+6.62 pp, 95% CI [+3.85, +9.40], McNemar b=8, c=39, p=5.53962e-06, Holm p=1.92211e-05
- CommonsenseQA / google/gemma-2-2b-it / exp2 -> exp3 / harmful_conformity: N=344, 0.0349 -> 0.0959, delta=+6.10 pp, 95% CI [+3.20, +9.30], McNemar b=4, c=25, p=0.000103716, Holm p=0.000311147
- CommonsenseQA / google/gemma-2-2b-it / exp2 -> exp3 / beneficial_revision: N=156, 0.0064 -> 0.0321, delta=+2.56 pp, 95% CI [+0.64, +5.13], McNemar b=0, c=4, p=0.125, Holm p=1
- MMLU / google/gemma-2-2b-it / exp1 -> exp2 / accuracy: N=499, 0.5511 -> 0.5451, delta=-0.60 pp, 95% CI [-1.60, +0.40], McNemar b=5, c=2, p=0.453125, Holm p=0.453125
- MMLU / google/gemma-2-2b-it / exp1 -> exp2 / target_adoption: N=499, 0.1563 -> 0.1663, delta=+1.00 pp, 95% CI [-0.20, +2.20], McNemar b=2, c=7, p=0.179688, Holm p=0.179688
- MMLU / google/gemma-2-2b-it / exp1 -> exp2 / change_rate: N=499, 0.0220 -> 0.0240, delta=+0.20 pp, 95% CI [-0.80, +1.20], McNemar b=3, c=4, p=1, Holm p=1
- MMLU / google/gemma-2-2b-it / exp1 -> exp2 / conformity_rate: N=428, 0.0164 -> 0.0280, delta=+1.17 pp, 95% CI [-0.23, +2.57], McNemar b=2, c=7, p=0.179688, Holm p=0.179688
- MMLU / google/gemma-2-2b-it / exp1 -> exp2 / harmful_conformity: N=278, 0.0180 -> 0.0216, delta=+0.36 pp, 95% CI [-1.08, +1.80], McNemar b=2, c=3, p=1, Holm p=1
- MMLU / google/gemma-2-2b-it / exp1 -> exp2 / beneficial_revision: N=221, 0.0136 -> 0.0000, delta=-1.36 pp, 95% CI [-3.17, +0.00], McNemar b=3, c=0, p=0.25, Holm p=1
- MMLU / google/gemma-2-2b-it / exp2 -> exp3 / accuracy: N=499, 0.5451 -> 0.4810, delta=-6.41 pp, 95% CI [-8.62, -4.21], McNemar b=34, c=2, p=1.94123e-08, Holm p=9.70613e-08
- MMLU / google/gemma-2-2b-it / exp2 -> exp3 / target_adoption: N=499, 0.1663 -> 0.2445, delta=+7.82 pp, 95% CI [+5.21, +10.42], McNemar b=5, c=44, p=7.59716e-09, Holm p=3.79858e-08
- MMLU / google/gemma-2-2b-it / exp2 -> exp3 / change_rate: N=499, 0.0240 -> 0.1283, delta=+10.42 pp, 95% CI [+7.82, +13.23], McNemar b=1, c=53, p=6.10623e-15, Holm p=3.05311e-14
- MMLU / google/gemma-2-2b-it / exp2 -> exp3 / conformity_rate: N=428, 0.0280 -> 0.1215, delta=+9.35 pp, 95% CI [+6.31, +12.38], McNemar b=4, c=44, p=1.51383e-09, Holm p=7.56916e-09
- MMLU / google/gemma-2-2b-it / exp2 -> exp3 / harmful_conformity: N=278, 0.0216 -> 0.1223, delta=+10.07 pp, 95% CI [+6.47, +14.03], McNemar b=1, c=29, p=5.7742e-08, Holm p=2.8871e-07
- MMLU / google/gemma-2-2b-it / exp2 -> exp3 / beneficial_revision: N=221, 0.0000 -> 0.0090, delta=+0.90 pp, 95% CI [+0.00, +2.26], McNemar b=0, c=2, p=0.5, Holm p=1
