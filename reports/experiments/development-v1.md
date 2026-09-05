# Development protocol V1

BUS-BRA and Curated BUSI are used as source domains for training and source-domain validation. BUS-UCLM is used as an out-of-distribution development domain, providing feedback on whether modelling and preprocessing choices improve generalization beyond the source distributions rather than only source-domain performance. Because this feedback is used during development, BUS-UCLM is not treated as a final unseen test set. BrEaST is therefore kept as a locked external test domain and is evaluated only after the final pipeline has been selected.

| Data | Role | Used for training? | Purpose |
|---|---|---:|---|
| BUS-BRA source train | Source training | yes | Learn the segmentation task from a source domain |
| Curated BUSI source train | Source training | yes | Learn from a second source domain |
| BUS-BRA + Curated BUSI source validation | In-domain validation (`source_val`) | no | Measure learning and overfitting on the source domains |
| BUS-UCLM | OOD development domain | no | Compare how well development decisions transfer to a different domain |
| BrEaST | Locked final external test domain | no | Provide the final evaluation after development decisions are fixed |

## Two validation views

`source_val` answers whether the model has learned the segmentation task on data drawn from its training domains. It is used to monitor ordinary source-domain performance and overfitting.

BUS-UCLM answers whether the learned solution transfers to a different acquisition environment. It is not an additional part of `source_val` and is never used for training. Because its results may guide architecture, preprocessing, augmentation, and model-selection decisions during development, BUS-UCLM is an **OOD development domain**, not an unbiased final test set.

For a domain-generalization experiment, a model with slightly lower `source_val` performance may be preferable when it performs substantially better on BUS-UCLM, provided that source performance remains acceptable and the selection rule is applied consistently. Source and OOD results should always be reported separately so that this trade-off remains visible.

BrEaST remains locked throughout development. It is evaluated only after the experimental choices have been finalized, preserving it as the estimate of generalization to a domain that did not influence those choices. BrEaST is particularly useful for final evaluation because it provides one image per patient, giving a comparatively strong patient-independent external test set.

## Why BUS-UCLM is suitable for OOD development

The [cross-domain EDA](../eda/image-domain-statistics.md) shows that BUS-UCLM differs materially from BUS-BRA and Curated BUSI through:

- a fixed 856 x 606 image geometry and characteristic scanner-export layout;
- different brightness and contrast distributions;
- recurring scanner graphics, measurement markers, and occasional Color Doppler;
- a substantially different class composition.

Normal scans represent 61.2% of BUS-UCLM, compared with approximately 2.8% in the combined BUS-BRA and Curated BUSI source data under this protocol. BUS-UCLM therefore tests not only appearance shift but also a strong lesion-prevalence shift. Performance on normal scans must be reported separately to expose false-positive segmentation that lesion-only Dice scores would miss.

## Source split limitation

BUS-BRA is partitioned at patient level to keep each patient entirely within either source train or source validation. Reliable patient identifiers are unavailable for Curated BUSI; therefore, its source train/validation partition is performed at image level. Potential same-patient overlap between these partitions cannot be excluded.

The machine-readable configuration for this protocol is stored in `src/ultrasound_dg/configs/development/v1.yaml`.
