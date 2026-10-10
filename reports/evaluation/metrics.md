# Evaluation metrics

The evaluation separates lesion segmentation quality from false-positive
behaviour on normal scans. This is important because the datasets contain very
different proportions of normal images, and a single pooled Dice score can hide
which failure mode changed.

Model outputs are logits. Evaluation first applies a sigmoid and then converts
each pixel to foreground when its probability is at least the configured
decision threshold (`0.5` in the current experiments). All metrics below are
computed from these binary predictions, except for the reported loss.

## Reported metrics

| Metric | Evaluated on | Meaning |
|---|---|---|
| `loss` | All images | Mean BCE + soft-Dice loss. It is useful for monitoring optimization, but is not the primary performance measure. |
| `dice` | All images | Mean image-level Dice across lesion-positive and normal scans. |
| `lesion_dice` | Lesion-positive images | Mean overlap between the predicted and reference lesion masks. |
| `lesion_iou` | Lesion-positive images | Mean intersection divided by union; it is stricter than Dice. |
| `lesion_precision` | Lesion-positive images | Of all pixels predicted as lesion, the fraction that is truly lesion. Lower values indicate over-segmentation. |
| `lesion_recall` | Lesion-positive images | Of all reference lesion pixels, the fraction recovered by the model. Lower values indicate under-segmentation. |
| `lesion_miss_rate` | Lesion-positive images | Fraction of scans for which the model predicts a completely empty mask despite a lesion being present. |
| `normal_fp_fraction` | Normal images | For each normal scan, the fraction of all pixels incorrectly predicted as lesion, averaged across normal scans. This measures the average **size** of false-positive regions. |
| `normal_fp_image_rate` | Normal images | Fraction of normal scans whose predicted lesion area is greater than `0.1%` of the image. This measures how **frequently** a non-trivial false positive occurs. |

For example, a model can have a high `normal_fp_image_rate` but a low
`normal_fp_fraction` when it produces small false-positive regions on many
normal scans. Conversely, fewer but very large false-positive regions can give
a lower image rate and a higher area fraction. The two metrics therefore answer
different questions and should be reported together.

`dice` across all images requires particular care. An empty prediction on a
normal image receives Dice equal to 1, while any non-empty prediction against an
empty reference mask receives a score close to 0. Consequently, `dice` depends
strongly on normal-case prevalence and can change even when lesion segmentation
quality does not. Cross-domain interpretation should therefore prioritize
`lesion_dice` together with the two normal false-positive metrics.

## Aggregation levels

Image-level metrics give every scan equal weight. `macro_source_lesion_dice`
first computes `lesion_dice` separately for each source domain and then averages
the domain results, so BUS-BRA and BUSI receive equal weight despite their
different sizes. This metric is used for tuning and as the primary checkpoint
selection criterion.

Patient-macro metrics use a two-stage average:

1. average the relevant image-level score across a patient's scans;
2. average the resulting patient scores, giving every patient equal weight.

Lesion patient metrics include only lesion-positive scans, while normal
false-positive patient metrics include only normal scans. Patient keys combine
`source_domain` and `patient_id` to avoid collisions between datasets. If the
required patient identifiers are unavailable, the patient-macro metric is
reported as `N/A`; `patient_id_coverage` shows the fraction of evaluable samples
with an identifier.

A patient may contribute scans with different diagnoses, as occurs frequently
in BUS-UCLM. The evaluation does not assign such a patient to one exclusive
diagnosis group: each scan enters the appropriate lesion or normal calculation,
after which scores are averaged within that patient. Patient-level diagnosis
counts must therefore not be interpreted as disjoint groups.

## Interpretation

No single metric fully describes segmentation performance:

- Dice and IoU measure spatial overlap;
- precision and recall distinguish over- from under-segmentation;
- lesion miss rate detects complete failures that mean overlap can obscure;
- normal false-positive metrics test whether the model can correctly return an
  empty mask;
- patient-macro metrics prevent patients with many scans from dominating the
  result.

Results are reported separately for source validation and the BUS-UCLM OOD
development domain. BrEaST remains reserved for final external evaluation and
is not used for model or checkpoint selection.
