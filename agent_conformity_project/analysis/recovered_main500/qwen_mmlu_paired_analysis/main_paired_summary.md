# Main protocol paired analysis

Bootstrap repetitions: 10000
Bootstrap seed: 12345
Multiplicity family: all prespecified model-dataset cells and protocol contrasts, corrected separately within each outcome

All deltas below are condition 2 minus condition 1.

- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / accuracy: N=500, 0.4020 -> 0.3040, delta=-9.80 pp, 95% CI [-13.80, -5.80], McNemar b=79, c=30, p=2.95585e-06, Holm p=2.95585e-06
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / target_adoption: N=500, 0.4940 -> 0.6080, delta=+11.40 pp, 95% CI [+6.60, +16.00], McNemar b=46, c=103, p=3.45107e-06, Holm p=3.45107e-06
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / change_rate: N=500, 0.3860 -> 0.4840, delta=+9.80 pp, 95% CI [+5.20, +14.60], McNemar b=51, c=100, p=8.21285e-05, Holm p=8.21285e-05
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / conformity_rate: N=433, 0.4226 -> 0.5520, delta=+12.93 pp, 95% CI [+7.39, +18.24], McNemar b=46, c=102, p=4.80527e-06, Holm p=4.80527e-06
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / harmful_conformity: N=316, 0.3608 -> 0.5222, delta=+16.14 pp, 95% CI [+10.13, +22.47], McNemar b=28, c=79, p=8.42001e-07, Holm p=8.42001e-07
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / beneficial_revision: N=184, 0.0163 -> 0.0054, delta=-1.09 pp, 95% CI [-2.72, +0.00], McNemar b=2, c=0, p=0.5, Holm p=1
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / accuracy: N=500, 0.3040 -> 0.6040, delta=+30.00 pp, 95% CI [+25.80, +34.20], McNemar b=3, c=153, p=1.38568e-41, Holm p=2.77136e-41
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / target_adoption: N=500, 0.6080 -> 0.1760, delta=-43.20 pp, 95% CI [-47.60, -38.80], McNemar b=220, c=4, p=7.71424e-60, Holm p=1.54285e-59
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / change_rate: N=500, 0.4840 -> 0.0420, delta=-44.20 pp, 95% CI [-48.60, -39.60], McNemar b=223, c=2, p=9.43103e-64, Holm p=1.88621e-63
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / conformity_rate: N=433, 0.5520 -> 0.0485, delta=-50.35 pp, 95% CI [-54.97, -45.50], McNemar b=220, c=2, p=7.34542e-63, Holm p=1.46908e-62
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / harmful_conformity: N=316, 0.5222 -> 0.0443, delta=-47.78 pp, 95% CI [-53.48, -42.09], McNemar b=153, c=2, p=5.29472e-43, Holm p=1.05894e-42
- MMLU / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / beneficial_revision: N=184, 0.0054 -> 0.0000, delta=-0.54 pp, 95% CI [-1.63, +0.00], McNemar b=1, c=0, p=1, Holm p=1
