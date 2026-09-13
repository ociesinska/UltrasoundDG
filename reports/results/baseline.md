# Untuned ERM baseline results

The initial ERM baseline is a U-Net with an ImageNet-pretrained ResNet34 encoder, trained on the BUS-BRA and Curated BUSI source domains. The best source-validation checkpoint was selected at epoch 18. BUS-UCLM was used only as the OOD development domain; the locked BrEaST test domain was not evaluated.

## Baseline performance and domain shift

The model achieved strong but uneven performance across the two source domains. On BUS-BRA, it reached a lesion Dice of **0.8862**, whereas performance on BUSI was substantially lower at **0.7500**. The corresponding macro-averaged source lesion Dice was therefore **0.8181**.

The macro average is used as the primary source-domain reference because it gives equal weight to both source datasets. In contrast, the pooled source lesion Dice of **0.8630** is dominated by the considerably larger BUS-BRA dataset.

| Lesion-positive metric | BUS-BRA validation | BUSI validation | Macro source | BUS-UCLM OOD |
|---|---:|---:|---:|---:|
| Lesion Dice | 0.8862 | 0.7500 | 0.8181 | 0.6895 |
| Lesion precision | 0.8730 | 0.7724 | — | 0.7257 |
| Lesion recall | 0.9202 | 0.7779 | — | 0.7148 |
| Lesion IoU | — | — | — | 0.6046 |
| Complete lesion miss rate | 0.00% | 3.90% | — | 6.15% |

The difference between the source domains already indicates substantial cross-dataset variability. On BUS-BRA, lesion precision and recall reached **0.8730** and **0.9202**, respectively, with no completely missed lesion-positive scans. On BUSI, precision decreased to **0.7724**, recall to **0.7779**, and the lesion miss rate increased to **3.90%**. Although both datasets were included during training, the model generalized considerably better to BUS-BRA than to BUSI.

Performance decreased further on the held-out BUS-UCLM OOD development domain. Lesion Dice dropped to **0.6895**, a decrease of approximately **12.9 percentage points** relative to the macro source lesion Dice. Lesion IoU was **0.6046**, while precision and recall fell to **0.7257** and **0.7148**, respectively. The complete lesion miss rate increased to **6.15%**, compared with **0.66%** on the pooled source validation set. The domain shift therefore affects both segmentation accuracy and the model's ability to consistently detect lesion regions.

## Pooled source and OOD metrics

| Metric | Pooled source validation | BUS-UCLM OOD |
|---|---:|---:|
| Loss | 0.2577 | 0.8417 |
| Dice, all images | 0.8604 | 0.5258 |
| Lesion Dice | 0.8630 | 0.6895 |
| Lesion precision | 0.8559 | 0.7257 |
| Lesion recall | 0.8959 | 0.7148 |
| Lesion IoU | 0.7844 | 0.6046 |
| Complete lesion miss rate | 0.66% | 6.15% |
| Mean false-positive fraction on normal scans | 0.5363% | 2.6976% |
| Normal scans with a non-trivial false-positive region | 15.38% | 54.88% |

## Normal-scan robustness

Since BUS-BRA contains no normal cases, the source-domain normal-image metrics originate entirely from BUSI. On BUSI validation images, the model falsely segmented on average **0.54%** of the image area, and **15.38%** of normal scans contained a false-positive region larger than the predefined 0.1% area threshold. On BUS-UCLM, the mean false-positive area increased to **2.70%**, while the false-positive image rate increased to **54.88%**.

The OOD degradation is therefore not limited to poorer delineation of existing lesions. The model also becomes substantially less reliable at distinguishing normal tissue from lesion-like regions in the unseen domain.

The pooled OOD Dice of **0.5258** should be interpreted with caution because it combines lesion segmentation quality with the ability to correctly produce empty masks for normal images, while the proportion of normal scans differs considerably between datasets. Lesion-specific metrics and normal-case false-positive metrics are therefore reported separately and treated as more informative indicators of cross-domain robustness.

These results establish a clear source-to-OOD generalization gap for the initial ERM configuration and provide a reference for subsequent hyperparameter tuning and domain-generalization experiments. They should not be interpreted as the final performance limit of the ERM baseline because its hyperparameters have not yet been systematically optimized.

Predictions were binarized at a probability threshold of `0.5`. A normal scan was counted as containing a non-trivial false-positive region when predicted lesion pixels occupied more than `0.1%` of the processed image.
