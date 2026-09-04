# Main protocol paired analysis

Bootstrap repetitions: 10000
Bootstrap seed: 12345
Multiplicity family: all prespecified model-dataset cells and protocol contrasts, corrected separately within each outcome

All deltas below are condition 2 minus condition 1.

- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / neutral_source_label -> model_source_label / accuracy: N=500, 0.0520 -> 0.6520, delta=+60.00 pp, 95% CI [+55.60, +64.40], McNemar b=3, c=303, p=7.32635e-86, Holm p=1.46527e-85
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / neutral_source_label -> model_source_label / target_adoption: N=500, 0.8840 -> 0.1920, delta=-69.20 pp, 95% CI [-73.20, -65.00], McNemar b=346, c=0, p=1.39525e-104, Holm p=2.7905e-104
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / neutral_source_label -> model_source_label / change_rate: N=500, 0.8680 -> 0.1360, delta=-73.20 pp, 95% CI [-77.00, -69.40], McNemar b=366, c=0, p=1.33061e-110, Holm p=2.66122e-110
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / neutral_source_label -> model_source_label / conformity_rate: N=467, 0.8779 -> 0.1370, delta=-74.09 pp, 95% CI [-77.94, -70.02], McNemar b=346, c=0, p=1.39525e-104, Holm p=2.7905e-104
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / neutral_source_label -> model_source_label / harmful_conformity: N=368, 0.8913 -> 0.1168, delta=-77.45 pp, 95% CI [-81.79, -73.10], McNemar b=285, c=0, p=3.21722e-86, Holm p=6.43445e-86
- CommonsenseQA / Qwen/Qwen2.5-3B-Instruct / neutral_source_label -> model_source_label / beneficial_revision: N=132, 0.0227 -> 0.0152, delta=-0.76 pp, 95% CI [-3.79, +2.27], McNemar b=3, c=2, p=1, Holm p=1
- CommonsenseQA / google/gemma-2-2b-it / neutral_source_label -> model_source_label / accuracy: N=500, 0.2200 -> 0.6200, delta=+40.00 pp, 95% CI [+35.60, +44.20], McNemar b=1, c=201, p=6.31636e-59, Holm p=6.31636e-59
- CommonsenseQA / google/gemma-2-2b-it / neutral_source_label -> model_source_label / target_adoption: N=500, 0.7240 -> 0.1680, delta=-55.60 pp, 95% CI [-59.80, -51.20], McNemar b=278, c=0, p=4.11805e-84, Holm p=4.11805e-84
- CommonsenseQA / google/gemma-2-2b-it / neutral_source_label -> model_source_label / change_rate: N=500, 0.6720 -> 0.1300, delta=-54.20 pp, 95% CI [-58.60, -49.80], McNemar b=272, c=1, p=3.6107e-80, Holm p=3.6107e-80
- CommonsenseQA / google/gemma-2-2b-it / neutral_source_label -> model_source_label / conformity_rate: N=468, 0.7051 -> 0.1111, delta=-59.40 pp, 95% CI [-63.68, -54.91], McNemar b=278, c=0, p=4.11805e-84, Holm p=4.11805e-84
- CommonsenseQA / google/gemma-2-2b-it / neutral_source_label -> model_source_label / harmful_conformity: N=345, 0.6696 -> 0.0986, delta=-57.10 pp, 95% CI [-62.32, -51.88], McNemar b=197, c=0, p=9.95682e-60, Holm p=9.95682e-60
- CommonsenseQA / google/gemma-2-2b-it / neutral_source_label -> model_source_label / beneficial_revision: N=155, 0.0065 -> 0.0323, delta=+2.58 pp, 95% CI [+0.65, +5.16], McNemar b=0, c=4, p=0.125, Holm p=0.25
