# ERM baseline results

The ERM baseline is a U-Net with an ImageNet-pretrained ResNet34 encoder, trained on the BUS-BRA and Curated BUSI source domains. The baseline uses AdamW with a learning rate of `1e-4`, weight decay of `1e-4`, batch size 16, and a maximum of 40 epochs. The checkpoint with the highest macro-averaged lesion Dice across the two source-validation domains was selected at epoch 22.

BUS-UCLM was used only as the OOD development domain. The locked BrEaST test domain was not evaluated.

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

The selected configuration was retrained as `baseline_tuned_v1`. Direct evaluation showed that the original and selected configurations produced identical metrics at the reported precision:

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
