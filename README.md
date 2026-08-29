# UltrasoundDG

**Domain Generalization for Breast Ultrasound Lesion Segmentation**

Detailed descriptions of the experiments and the full analysis workflow are available in the [project reports](reports/eda/image-domain-statistics.md).

Breast ultrasound segmentation models are commonly trained and evaluated on data originating from the same dataset. In practice, however, ultrasound images can vary substantially across hospitals, scanners, acquisition protocols, and patient populations.

A model that performs well on its source dataset may therefore degrade when applied to data from a previously unseen clinical environment.

**UltrasoundDG** investigates this problem through **domain generalization**: training breast-lesion segmentation models using one or more source domains and evaluating their ability to generalize to unseen clinical domains without using target-domain data during training.

## Research questions

1. **How large is the domain shift?**
   How much does segmentation performance decrease when a model is evaluated on data from a different clinical source?

2. **Why are the domains different?**
   Can differences between datasets be identified through image statistics and learned feature representations?

3. **Can we reduce the generalization gap?**
   Can domain-generalization strategies improve segmentation performance on previously unseen clinical data?


## Planned methodology

The project will progressively compare:

* conventional single-source training as a baseline;
* multi-source training;
* ultrasound-specific domain randomization and augmentation;
* feature-level domain-generalization methods such as MixStyle;
* potentially domain-adversarial representation learning;
* cross-domain representation analysis;
* failure analysis and uncertainty estimation.

The emphasis is not on comparing large numbers of segmentation architectures, but on understanding and improving **generalization across clinical domains**.

## Data

The study uses independent public breast-ultrasound datasets collected in different clinical environments.

| Dataset                         | Role                              | Origin |
| ------------------------------- | --------------------------------- | ------ |
| **BUS-BRA**                     | Primary source domain             | Brazil |
| **Curated BUSI**                | Secondary source domain           | Egypt  |
| **BUS-UCLM**                    | External development benchmark    | Spain  |
| **Breast-Lesions-USG (BrEaST)** | Locked final external test domain | Poland |

BUS-BRA and Curated BUSI are used as source domains for model development.

BUS-UCLM is used to investigate external generalization during experimentation.

BrEaST is reserved as a **locked final external test set** and is not intended to be used for model selection or hyperparameter tuning.

Raw and processed medical imaging datasets are not distributed with this repository and are excluded from version control.

Detailed dataset descriptions, official download links, versions, licences, and citation information are available in [`DATASETS.md`](DATASETS.md).

## Experimental overview

The initial experimental setup follows the general structure:

```text
Source domains
BUS-BRA + BUSI
       │
       ▼
 segmentation model
       │
       ▼
 domain-generalization strategy
       │
       ▼
 unseen clinical domain
 BUS-UCLM / BrEaST
```

The main quantity of interest is the **generalization gap** between in-domain and external-domain performance.

Evaluation will include segmentation metrics such as:

* Dice score
* IoU / Jaccard
* precision
* recall / sensitivity
* HD95

Additional analyses will investigate performance across lesion characteristics, dataset-specific image properties, and model failure cases.

## Reports

* [Cross-domain EDA](reports/eda/image-domain-statistics.md)

## Licence

Source code developed as part of UltrasoundDG is released under the MIT License.

The external medical datasets used by the project are **not covered by the repository's MIT licence** and remain subject to their respective licences, copyright terms, and attribution requirements.

See [`DATASETS.md`](DATASETS.md) for details.
