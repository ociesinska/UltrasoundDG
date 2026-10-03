# ERM baseline results

The ERM baseline is a U-Net with an ImageNet-pretrained ResNet34 encoder, trained on the BUS-BRA and Curated BUSI source domains. The baseline uses AdamW with a learning rate of `1e-4`, weight decay of `1e-4`, batch size 16, and a maximum of 40 epochs. The checkpoint with the highest macro-averaged lesion Dice across the two source-validation domains was selected at epoch 22.

BUS-UCLM was used only as the OOD development domain. Model performance on the reserved BrEaST external evaluation domain was not evaluated.

## Baseline performance and domain shift

Performance differed substantially between the two source domains. The model reached a lesion Dice of **0.8885** on BUS-BRA and **0.7798** on BUSI, giving a macro source lesion Dice of **0.8341**.

The macro average is used as the primary lesion-segmentation reference because it gives equal weight to both source datasets. In contrast, the pooled source lesion Dice of **0.8700** is dominated by the considerably larger BUS-BRA dataset.

| Lesion-positive metric | BUS-BRA validation | BUSI validation | Macro source | BUS-UCLM OOD |
|---|---:|---:|---:|---:|
| Lesion Dice | 0.8885 | 0.7798 | 0.8341 | 0.7109 |
| Lesion precision | 0.8998 | 0.8142 | — | 0.7982 |
| Lesion recall | 0.8969 | 0.7885 | — | 0.6841 |
| Lesion IoU | 0.8136 | 0.6861 | — | 0.6209 |
| Complete lesion miss rate | 0.00% | 1.30% | — | 5.00% |

Although both source datasets were included during training, the model generalized considerably better to BUS-BRA than to BUSI. The difference of approximately 10.9 percentage points in lesion Dice demonstrates that pooling source datasets does not result in equally strong performance across them.

Performance decreased further on BUS-UCLM. Lesion Dice was **0.7109**, approximately 12.3 percentage points below the macro source lesion Dice. Precision remained relatively high at **0.7982**, while recall decreased to **0.6841** and the complete lesion miss rate increased to **5.00%**. The OOD model therefore behaves more conservatively and misses lesion regions more frequently.

## Pooled source and OOD metrics

| Metric | Pooled source validation | BUS-UCLM OOD |
|---|---:|---:|
| Loss | 0.2556 | 0.8272 |
| Dice, all images | 0.8564 | 0.5266 |
| Lesion Dice | 0.8700 | 0.7109 |
| Lesion precision | 0.8852 | 0.7982 |
| Lesion recall | 0.8785 | 0.6841 |
| Lesion IoU | 0.7919 | 0.6209 |
| Complete lesion miss rate | 0.22% | 5.00% |
| Mean false-positive fraction on normal scans | 1.9352% | 1.9121% |
| Normal scans with a non-trivial false-positive region | 53.85% | 52.20% |

## Normal-scan robustness

BUS-BRA contains no normal validation cases, so the source normal-image metrics originate entirely from BUSI. The mean falsely segmented area was similar on BUSI and BUS-UCLM normal scans: **1.94%** and **1.91%**, respectively. A non-trivial false-positive region was present in **53.85%** of BUSI normal scans and **52.20%** of BUS-UCLM normal scans.

The normal-case problem is therefore not uniquely caused by the OOD shift: the baseline already produces frequent false-positive regions on normal source images. This behaviour is particularly important for BUS-UCLM because normal scans constitute a much larger share of that domain.

The pooled OOD Dice of **0.5266** should be interpreted with caution because it combines lesion segmentation quality with the ability to correctly produce empty masks for normal images, while the proportion of normal scans differs considerably between datasets. Lesion-specific metrics and normal-case false-positive metrics are therefore reported separately.

## First hyperparameter tuning round

A small Optuna grid search was performed over four learning rates (`1e-3`, `3e-4`, `1e-4`, and `3e-5`) and four weight-decay values (`0`, `1e-5`, `1e-4`, and `1e-3`). Trials were trained for up to 40 epochs and ranked using macro source lesion Dice. BUS-UCLM was not used for hyperparameter selection. Of the 16 initiated trials, 9 completed, 6 were pruned, and 1 failed before reporting a validation result.

The best objective value was obtained with `learning_rate=1e-4`. Weight decay values of `0`, `1e-5`, and `1e-4` all produced exactly the same macro source lesion Dice of **0.8341** at epoch 22. Optuna selected `weight_decay=0` because it was the first of the tied configurations, not because it achieved a better score than the original `1e-4` value.

The selected configuration was retrained as `baseline_multisource_tuned_v1`. Direct evaluation showed that the original and selected configurations produced identical metrics at the reported precision:

| Metric | Original configuration (`wd=1e-4`) | Optuna-selected configuration (`wd=0`) |
|---|---:|---:|
| Best epoch | 22 | 22 |
| Macro source lesion Dice | 0.8341 | 0.8341 |
| BUS-BRA lesion Dice | 0.8885 | 0.8885 |
| BUSI lesion Dice | 0.7798 | 0.7798 |
| Pooled source lesion Dice | 0.8700 | 0.8700 |
| BUS-UCLM lesion Dice | 0.7109 | 0.7109 |
| BUS-UCLM lesion precision | 0.7982 | 0.7982 |
| BUS-UCLM lesion recall | 0.6841 | 0.6841 |
| BUS-UCLM complete lesion miss rate | 5.00% | 5.00% |
| BUS-UCLM normal false-positive fraction | 1.9121% | 1.9121% |
| BUS-UCLM normal false-positive image rate | 52.20% | 52.20% |

The first tuning round therefore did not improve the ERM baseline. Instead, it confirmed that the original learning rate was already the best choice within the tested range and that small weight-decay values had no measurable effect on the thresholded predictions. The original `weight_decay=1e-4` remains a valid baseline choice; the tuning result does not provide evidence for preferring `weight_decay=0`.

These results were obtained with one random seed (`42`). Multi-seed evaluation is required before treating small differences in future experiments as robust.

Predictions were binarized at a probability threshold of `0.5`. A normal scan was counted as containing a non-trivial false-positive region when predicted lesion pixels occupied more than `0.1%` of the processed image.

## Single-source versus multi-source training

To examine whether training on two source domains improves external generalization by itself, additional ERM baselines were trained using only BUS-BRA or only Curated BUSI. All three configurations used the same preprocessing, U-Net architecture with an ImageNet-pretrained ResNet34 encoder, learning rate of `1e-4`, maximum of 40 epochs, and decision threshold of `0.5`. The development split remained fixed with seed `42`, while model training was repeated with seeds `42`, `123`, and `456`. The table reports the mean and sample standard deviation across those three runs.

| Training domains | OOD Dice, all images | OOD lesion Dice | OOD lesion precision | OOD lesion recall | OOD lesion IoU |
|---|---:|---:|---:|---:|---:|
| BUS-BRA | 0.330 ± 0.032 | 0.708 ± 0.011 | 0.761 ± 0.011 | 0.732 ± 0.022 | 0.609 ± 0.007 |
| Curated BUSI | 0.308 ± 0.156 | 0.565 ± 0.092 | 0.550 ± 0.147 | 0.714 ± 0.082 | 0.457 ± 0.097 |
| BUS-BRA + Curated BUSI | **0.531 ± 0.065** | **0.723 ± 0.021** | **0.797 ± 0.006** | 0.713 ± 0.038 | **0.632 ± 0.019** |

| Training domains | Complete lesion miss rate | Normal FP fraction | Normal FP image rate |
|---|---:|---:|---:|
| BUS-BRA | **1.54 ± 2.00%** | 4.61 ± 1.43% | 86.59 ± 5.63% |
| Curated BUSI | 2.82 ± 4.24% | 4.18 ± 3.67% | 81.87 ± 27.29% |
| BUS-BRA + Curated BUSI | 4.36 ± 2.19% | **2.15 ± 0.69%** | **52.44 ± 11.83%** |

The BUS-BRA-only model provided stable lesion segmentation across seeds, reaching an OOD lesion Dice of `0.708 ± 0.011`. It nevertheless produced false-positive regions on most normal BUS-UCLM scans: the mean false-positive image rate was 86.59%. This consistent failure is compatible with the absence of normal examples in BUS-BRA training data.

The BUSI-only model was the least stable configuration. Its OOD lesion Dice ranged from `0.4623` to `0.6403`, while its false-positive image rate ranged from 50.49% to 100%. The apparently conservative behaviour observed for seed `42` did not reproduce for the other seeds. Its high variance demonstrates why conclusions from the initial single-seed comparison would have been misleading.

Multi-source training achieved the highest mean OOD lesion Dice, lesion precision, lesion IoU, and all-image Dice. Its advantage over BUS-BRA-only lesion Dice was modest (`0.723` versus `0.708`), but the normal-scan improvement was substantially larger: the mean false-positive fraction decreased from 4.61% to 2.15%, and the false-positive image rate decreased from 86.59% to 52.44%. Multi-source lesion precision was also highly stable across seeds (`0.797 ± 0.006`). These results support the conclusion that combining source domains produces a better-balanced and more robust external-domain model than either single-source configuration.

The complete lesion miss rate requires a separate interpretation. BUS-BRA-only achieved the lowest mean miss rate, whereas the multi-source model missed more entire lesion masks despite producing better average overlap and precision. Multi-source training therefore improves overall segmentation quality and normal-scan robustness, but does not dominate BUS-BRA-only training for every clinically relevant failure mode.

Source-validation scores are not directly comparable between configurations because each single-source model is evaluated on a different source distribution, while the multi-source model is selected using a macro average over both source domains. The OOD results are directly comparable because every model is evaluated on the same BUS-UCLM samples. With only three seeds, the reported variability describes the observed runs but is not intended as a formal statistical significance test.

## Domain-balanced source sampling

The natural multi-source baseline samples images uniformly from the pooled training set, causing the larger BUS-BRA domain to appear more frequently than Curated BUSI. A domain-balanced ERM variant instead assigns each training image a weight inversely proportional to the size of its source domain. This gives BUS-BRA and Curated BUSI equal expected sampling probability while preserving the original number of samples and optimization steps per epoch. Sampling is performed with replacement.

Natural and domain-balanced sampling were compared using the same split, preprocessing, architecture, loss, optimizer, hyperparameters, checkpoint-selection metric, and training seeds (`42`, `123`, and `456`). Only the training sampling strategy changed. The table reports mean and sample standard deviation across the three seeds.

| Metric | Natural sampling | Domain-balanced sampling |
|---|---:|---:|
| Macro source lesion Dice | **0.831 ± 0.008** | 0.825 ± 0.002 |
| BUS-BRA source lesion Dice | 0.889 ± 0.003 | **0.893 ± 0.004** |
| BUSI source lesion Dice | **0.774 ± 0.013** | 0.757 ± 0.003 |
| OOD Dice, all images | **0.531 ± 0.065** | 0.393 ± 0.123 |
| OOD lesion Dice | 0.723 ± 0.021 | **0.727 ± 0.023** |
| OOD lesion precision | **0.797 ± 0.006** | 0.770 ± 0.010 |
| OOD lesion recall | 0.713 ± 0.038 | **0.752 ± 0.033** |
| OOD lesion IoU | **0.632 ± 0.019** | 0.631 ± 0.022 |
| Complete lesion miss rate | 4.36 ± 2.19% | **0.77 ± 0.67%** |
| Normal FP fraction | **2.15 ± 0.69%** | 3.06 ± 0.71% |
| Normal FP image rate | **52.44 ± 11.83%** | 77.07 ± 20.42% |

Domain-balanced sampling produced a clear sensitivity-specific benefit. OOD lesion recall increased from `0.713` to `0.752`, and the complete lesion miss rate decreased from 4.36% to 0.77%. Mean lesion Dice changed only slightly (`0.723` to `0.727`), while lesion IoU remained effectively unchanged.

These gains came with a substantial loss of specificity. OOD lesion precision decreased, the mean falsely segmented fraction on normal scans increased from 2.15% to 3.06%, and the normal false-positive image rate increased from 52.44% to 77.07%. Consequently, all-image OOD Dice decreased markedly from `0.531` to `0.393`. Domain-balanced sampling also failed to improve BUSI source-validation Dice despite increasing the expected frequency of BUSI samples during training.

Under the present evaluation priorities, domain-balanced sampling is therefore not a better general-purpose replacement for natural pooled sampling. It creates a more lesion-sensitive model that rarely misses an entire lesion, but this is achieved by predicting lesion regions more aggressively and producing substantially more false positives on normal scans. This trade-off may also be reinforced by selecting checkpoints exclusively with macro source lesion Dice, which does not penalize false positives on normal images. The result shows that balancing source-domain frequency is not equivalent to balancing lesion prevalence or optimizing normal-case robustness. Natural sampling remains the preferred ERM baseline for subsequent comparisons, while domain-balanced sampling is retained as an informative sensitivity-oriented ablation.

## Checkpoint-selection protocol update

The results above were produced with the original checkpoint rule: select the epoch with the highest macro source lesion Dice. Natural and domain-balanced sampling used the same rule, so their comparison remains internally consistent. However, the observed sensitivity-specific trade-off showed that lesion Dice alone does not distinguish models with substantially different false-positive behaviour on normal scans.

The subsequent [Model selection and tuning V2 protocol](../experiments/model-selection-v2.md) retains lesion Dice as the primary criterion but uses source normal false-positive fraction to choose among checkpoints and tuning trials within `0.01` of the best macro source lesion Dice. BUS-UCLM remains excluded from this selection process. Baseline and domain-balanced models must be rerun under V2 before their new results can replace or be directly compared with the tables above.
