# Main protocol paired analysis

Bootstrap repetitions: 10000
Bootstrap seed: 12345
Multiplicity family: all prespecified model-dataset cells and protocol contrasts, corrected separately within each outcome

All deltas below are condition 2 minus condition 1.

- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / accuracy: N=500, 0.5660 -> 0.3260, delta=-24.00 pp, 95% CI [-28.40, -19.60], McNemar b=137, c=17, p=1.72872e-24, Holm p=1.72872e-24
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / target_adoption: N=500, 0.2920 -> 0.6140, delta=+32.20 pp, 95% CI [+27.60, +37.00], McNemar b=19, c=180, p=4.48412e-34, Holm p=4.48412e-34
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / change_rate: N=500, 0.2420 -> 0.5660, delta=+32.40 pp, 95% CI [+27.60, +37.00], McNemar b=19, c=181, p=2.47584e-34, Holm p=2.47584e-34
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / conformity_rate: N=468, 0.2436 -> 0.5876, delta=+34.40 pp, 95% CI [+29.49, +39.53], McNemar b=19, c=180, p=4.48412e-34, Holm p=4.48412e-34
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / harmful_conformity: N=368, 0.2283 -> 0.5571, delta=+32.88 pp, 95% CI [+27.45, +38.59], McNemar b=15, c=136, p=1.41708e-25, Holm p=1.41708e-25
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp1 -> exp2 / beneficial_revision: N=132, 0.0227 -> 0.0227, delta=+0.00 pp, 95% CI [-2.27, +2.27], McNemar b=1, c=1, p=1, Holm p=1
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / accuracy: N=500, 0.3260 -> 0.7120, delta=+38.60 pp, 95% CI [+34.40, +43.00], McNemar b=3, c=196, p=3.26982e-54, Holm p=6.53964e-54
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / target_adoption: N=500, 0.6140 -> 0.0920, delta=-52.20 pp, 95% CI [-56.60, -47.80], McNemar b=261, c=0, p=5.39761e-79, Holm p=1.07952e-78
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / change_rate: N=500, 0.5660 -> 0.0340, delta=-53.20 pp, 95% CI [-57.60, -48.80], McNemar b=266, c=0, p=1.68675e-80, Holm p=3.3735e-80
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / conformity_rate: N=468, 0.5876 -> 0.0299, delta=-55.77 pp, 95% CI [-60.26, -51.28], McNemar b=261, c=0, p=5.39761e-79, Holm p=1.07952e-78
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / harmful_conformity: N=368, 0.5571 -> 0.0299, delta=-52.72 pp, 95% CI [-57.88, -47.55], McNemar b=194, c=0, p=7.96546e-59, Holm p=1.59309e-58
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / exp2 -> exp3 / beneficial_revision: N=132, 0.0227 -> 0.0076, delta=-1.52 pp, 95% CI [-4.55, +1.52], McNemar b=3, c=1, p=0.625, Holm p=1
