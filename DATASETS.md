# Datasets

UltrasoundDG uses several independently collected public breast-ultrasound datasets to study segmentation performance under clinical domain shift.

The datasets are **not redistributed through this repository**. They must be downloaded from their respective official sources and remain subject to their original licences, copyright terms, and citation requirements.

---

## BUS-BRA

**Role in UltrasoundDG:** Primary source domain

**Dataset:** BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems

**Version:** 1.0

**Official source:**
https://zenodo.org/records/8231412

**DOI:**
https://doi.org/10.5281/zenodo.8231412

BUS-BRA contains breast ultrasound images from patients evaluated at the Brazilian National Cancer Institute and includes lesion delineations and clinical information such as BI-RADS categories.

The dataset contains:

* 1,875 ultrasound images;
* 1,064 patients;
* benign and malignant lesions;
* expert lesion delineations;
* data acquired using multiple ultrasound systems.

### Licence and copyright

The official Zenodo record identifies the Program of Biomedical Engineering of the Federal University of Rio de Janeiro (PEB/COPPE-UFRJ) as the copyright holder and principal distributor.

The official record requires publications using BUS-BRA to cite the associated work. At the time this repository was created, the Zenodo record did not specify a standard open-source/data licence such as CC BY.

BUS-BRA data should therefore **not be assumed to be covered by the MIT licence of this repository**.

### Citation

Gómez-Flores, W., Gregorio-Calas, M. J., & Pereira, W. C. A.
*BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems.*
Medical Physics, 51, 3110–3123, 2024.

DOI:
https://doi.org/10.1002/mp.16812

---

## Curated BUSI

**Role in UltrasoundDG:** Secondary source domain

**Dataset:** Curated BUSI Dataset — Curated Breast Ultrasound Images

**Version:** 1.0

**Official source:**
https://zenodo.org/records/19047974

**DOI:**
https://doi.org/10.5281/zenodo.19047974

Curated BUSI is a cleaned version of the Breast Ultrasound Images Dataset (BUSI).

The original BUSI dataset was collected in Egypt and contains ultrasound images classified as:

* benign;
* malignant;
* normal.

Lesion segmentation masks are available for lesion-containing images.

The curated release addresses problems identified in the original dataset, including duplicate and problematic samples.

### Licence

**Creative Commons Attribution 4.0 International — CC BY 4.0**

Users of the dataset must provide appropriate attribution in accordance with the licence terms.

### Associated publication

Aumente-Maestro, C., Díez, J., & Remeseiro, B.
*A multi-task framework for breast cancer segmentation and classification in ultrasound imaging.*
Computer Methods and Programs in Biomedicine, 260, 108540, 2025.

DOI:
https://doi.org/10.1016/j.cmpb.2024.108540

### Original BUSI dataset

Original BUSI publication:

Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A.
*Dataset of Breast Ultrasound Images.*
Data in Brief, 28, 104863, 2020.

DOI:
https://doi.org/10.1016/j.dib.2019.104863

---

## BUS-UCLM

**Role in UltrasoundDG:** External development benchmark

**Dataset:** BUS-UCLM: Breast Ultrasound Lesion Segmentation Dataset

**Version:** 3

**Official source:**
https://data.mendeley.com/datasets/7fvgj4jsp7/3

**DOI:**
https://doi.org/10.17632/7fvgj4jsp7.3

BUS-UCLM was collected at Ciudad Real General University Hospital in Spain.

The dataset contains:

* 683 ultrasound images;
* 38 patients;
* 174 benign images;
* 90 malignant images;
* 419 normal images;
* expert lesion segmentation masks.

Images were acquired using a Siemens ACUSON S2000 ultrasound system.

The dataset preserves anonymized patient identifiers, allowing images belonging to the same patient to be identified and preventing patient-level leakage during evaluation.

### Licence

**Creative Commons Attribution 4.0 International — CC BY 4.0**

Users must provide appropriate attribution to the dataset creators.

### Use within UltrasoundDG

BUS-UCLM is treated as an **external domain**.

It is not used as a source domain during the primary domain-generalization training experiments.

It may be used during method development to evaluate whether changes improve external generalization.

---

## Breast-Lesions-USG / BrEaST

**Role in UltrasoundDG:** Locked final external test domain

**Dataset:** Breast-Lesions-USG — A Curated Benchmark Dataset for Ultrasound Based Breast Lesion Analysis

**Official source:**
https://www.cancerimagingarchive.net/collection/breast-lesions-usg/

**Dataset DOI:**
https://doi.org/10.7937/9WKK-Q141

The dataset is distributed through **The Cancer Imaging Archive (TCIA)** and was collected in Poland.

It contains:

* 256 patients;
* 256 ultrasound examinations;
* 266 segmented lesions;
* benign and malignant cases;
* segmentation annotations;
* accompanying clinical information;
* diagnosis and follow-up information.

Both imaging/segmentation data and accompanying clinical data are provided.

### Licence

**Creative Commons Attribution 4.0 International — CC BY 4.0**

Users must provide appropriate attribution according to TCIA and dataset requirements.

### Citation

Pawłowska, A. et al.
*A Curated benchmark dataset for ultrasound based breast lesion analysis.*
Scientific Data, 11, 148, 2024.

DOI:
https://doi.org/10.1038/s41597-024-02984-z

### Use within UltrasoundDG

BrEaST is reserved as the **final external evaluation dataset**.

Its results should not be used to:

* select model architectures;
* tune hyperparameters;
* select augmentations;
* choose domain-generalization methods;
* tune post-processing.

The intention is to use BrEaST only after the modelling choices have been finalized, providing a cleaner estimate of generalization to an independently collected clinical dataset.

---

# Local data structure

The expected local organization is:

```text
data/
├── raw/
│   ├── bus_bra/
│   ├── busi/
│   ├── bus_uclm/
│   └── breast_usg/
│
├── processed/
└── manifests/
```

The contents of `data/raw/` and `data/processed/` are excluded from version control.

---

# Repository licence vs dataset licences

The MIT licence included with UltrasoundDG applies only to original source code developed for this repository.

It does **not** grant any rights to the datasets described above.

Each dataset remains governed by its respective licence, copyright terms, citation requirements, and data-use conditions.
