# Final ERM baseline

The final reference baseline is a U-Net with an ImageNet-pretrained ResNet34
encoder, trained jointly on BUS-BRA and Curated BUSI using natural sampling.
Its tracked configuration is
[`baseline_multisource_tuned_v2.yaml`](../../src/ultrasound_dg/configs/experiments/baseline/baseline_multisource_tuned_v2.yaml).
Metric definitions and aggregation rules are described in the
[evaluation metrics guide](../evaluation/metrics.md).

| Setting | Value |
|---|---|
| Source domains | BUS-BRA + Curated BUSI |
| Architecture | U-Net, ResNet34 encoder |
| Encoder initialization | ImageNet |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-3` |
| Batch size | 16 |
| Maximum epochs | 40 |
| Sampling | Natural pooled sampling |
| Decision threshold | 0.5 |
| Augmentation | None |

The learning rate and weight decay were selected by V2 tuning. For every seed,
the checkpoint-selection rule first retained epochs within `0.01` of the best
macro source lesion Dice and then preferred the lowest false-positive fraction
on normal source-validation images. Hyperparameter and checkpoint selection
used source validation only. BUS-UCLM was used for OOD development evaluation,
and BrEaST was not evaluated.

## Results

Training was repeated with seeds `42`, `123`, and `456`. Values are mean ±
sample standard deviation across the three runs.

| Metric | Result |
|---|---:|
| BUS-BRA source lesion Dice | 0.8920 ± 0.0027 |
| BUSI source lesion Dice | 0.7578 ± 0.0066 |
| Macro source lesion Dice | 0.8249 ± 0.0032 |
| BUS-UCLM Dice, all images | 0.5411 ± 0.0877 |
| BUS-UCLM lesion Dice | 0.7149 ± 0.0277 |
| BUS-UCLM lesion precision | 0.7745 ± 0.0374 |
| BUS-UCLM lesion recall | 0.7127 ± 0.0271 |
| BUS-UCLM lesion IoU | 0.6275 ± 0.0219 |
| BUS-UCLM complete lesion miss rate | 5.26 ± 2.91% |
| BUS-UCLM normal FP fraction | 1.93 ± 0.21% |
| BUS-UCLM normal FP image rate | 50.73 ± 12.89% |
| BUS-UCLM patient-macro lesion Dice | 0.6958 ± 0.0294 |

The source-to-OOD lesion Dice gap is approximately 0.11. Patient-macro Dice is
also lower than image-level Dice, showing that the OOD result is not explained
only by patients with many scans. Normal-scan robustness remains an important
failure mode: although the falsely segmented area is usually small, about half
of normal BUS-UCLM scans contain a predicted region larger than the predefined
0.1% image-area threshold.

The source-domain result is uneven: BUS-BRA lesion Dice exceeds BUSI Dice by
approximately 0.13. Multi-source training therefore does not remove the
difference in source-domain difficulty.

## Source and sampling ablations

All variants used the same architecture, preprocessing, tuned optimizer
settings, decision threshold, V2 checkpoint selection, and training seeds.
Only the training domains or sampling strategy changed.

| Training setup | OOD all-image Dice | OOD lesion Dice | OOD recall | Miss rate | Normal FP fraction / image rate |
|---|---:|---:|---:|---:|---:|
| **Multi-source, natural** | **0.541 ± 0.088** | **0.715 ± 0.028** | 0.713 ± 0.027 | 5.26 ± 2.91% | **1.93 ± 0.21% / 50.73 ± 12.89%** |
| Multi-source, domain-balanced | 0.497 ± 0.051 | 0.705 ± 0.030 | 0.725 ± 0.049 | 4.62 ± 1.68% | 2.16 ± 0.50% / 58.21 ± 9.00% |
| BUS-BRA only | 0.306 ± 0.053 | 0.698 ± 0.013 | **0.741 ± 0.033** | **0.64 ± 1.11%** | 5.30 ± 1.29% / 91.54 ± 10.30% |
| BUSI only | 0.355 ± 0.143 | 0.604 ± 0.012 | 0.697 ± 0.052 | 3.08 ± 3.67% | 2.26 ± 1.21% / 73.90 ± 24.75% |

Natural multi-source training gives the best overall balance of lesion overlap,
precision, patient-level performance, and normal-scan robustness. Domain
balancing provides a small recall and miss-rate benefit but worsens Dice, IoU,
precision, and false-positive behaviour, so it remains an ablation rather than
the reference baseline.

BUS-BRA-only training rarely misses an entire lesion but predicts lesions on
almost every normal BUS-UCLM scan. This is consistent with BUS-BRA providing no
normal training examples. BUSI-only training is substantially weaker on OOD
lesion segmentation. Together, these results support multi-source training as
the reference ERM setup.

## Why BUSI-only normal-scan results vary across seeds

The largest BUSI-only variation is not in lesion Dice, which is relatively
stable at `0.604 ± 0.012`. It is concentrated in predictions on normal scans:
the OOD normal FP image rate is `73.90 ± 24.75%`, ranging from 45.37% to 89.51%,
and consequently all-image Dice is also highly variable.

Several factors plausibly contribute:

- BUSI is the smaller source dataset and contains relatively few normal
  examples, so stochastic optimization can move the normal/lesion decision
  boundary substantially between runs;
- only 13 normal BUSI images occur in source validation, making the secondary
  false-positive checkpoint criterion noisy;
- reliable patient identifiers are unavailable, and the image-level split may
  contain correlated scans, reducing the effective diversity of the data;
- a model trained on one acquisition style can rely on source-specific cues,
  making its normal predictions especially sensitive to BUS-UCLM appearance
  shift;
- the 0.5 pixel threshold and 0.1% image-level FP threshold turn small logit
  changes into a discrete change in whether an entire scan counts as a false
  positive.

The three-seed estimates describe the observed variability but are not a formal
significance analysis.

## Baseline decision

`baseline_multisource_tuned_v2` is the final ERM reference for subsequent
augmentation and domain-generalization experiments. Single-source and
domain-balanced configurations are retained as ablations. Future experiments
must compare against the same three-seed baseline and must not use BrEaST for
model or hyperparameter selection.
