import pandas as pd


def manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:

    summary = manifest.groupby("source_domain").agg(
        sample_count=("sample_id", "size"),
        patient_count=("patient_id", "nunique"),
        lesion_ratio=("has_lesion", "mean"),
    )

    diagnosis_counts = pd.crosstab(manifest["source_domain"], manifest["diagnosis"])

    diagnosis_counts = diagnosis_counts.reindex(
        columns=["normal", "benign", "malignant"], fill_value=0
    ).add_suffix("_count")

    summary = summary.join(diagnosis_counts)

    diagnosis_ratios = pd.crosstab(
        manifest["source_domain"], manifest["diagnosis"], normalize="index"
    )

    diagnosis_ratios = diagnosis_ratios.reindex(
        columns=["normal", "benign", "malignant"],
        fill_value=0,
    ).add_suffix("_ratio")

    summary = summary.join(diagnosis_ratios)

    images_per_patient = (
        manifest.dropna(subset=["patient_id"])
        .groupby(["source_domain", "patient_id"])
        .size()
        .rename("image_count")
    )
    patient_image_stats = (
        images_per_patient.reset_index()
        .groupby("source_domain")
        .agg(
            min_images_per_patient=("image_count", "min"),
            mean_images_per_patient=("image_count", "mean"),
            median_images_per_patient=("image_count", "median"),
            max_images_per_patient=("image_count", "max"),
        )
    )
    summary = summary.join(patient_image_stats)

    multi_image_patients = (
        images_per_patient.gt(1)
        .groupby(level="source_domain")
        .sum()
        .rename("multi_image_patient_count")
    )
    summary = summary.join(multi_image_patients)

    patient_id_coverage = (
        manifest["patient_id"]
        .notna()
        .groupby(manifest["source_domain"])
        .mean()
        .rename("patient_id_coverage")
    )

    summary = summary.join(patient_id_coverage)

    missing_patient_id_count = (
        manifest["patient_id"]
        .isna()
        .groupby(manifest["source_domain"])
        .sum()
        .rename("missing_patient_id_count")
    )
    summary = summary.join(missing_patient_id_count)

    mask_available_ratio = (
        manifest["mask_path"]
        .notna()
        .groupby(manifest["source_domain"])
        .mean()
        .rename("mask_available_ratio")
    )
    summary = summary.join(mask_available_ratio)

    missing_mask_count = (
        manifest["mask_path"]
        .isna()
        .groupby(manifest["source_domain"])
        .sum()
        .rename("missing_mask_count")
    )
    summary = summary.join(missing_mask_count)

    return summary.reset_index()


def patient_diagnosis_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize potentially overlapping diagnoses at patient level."""

    patients = manifest.dropna(subset=["patient_id"])

    diagnosis_counts = (
        patients.groupby(["source_domain", "diagnosis"])["patient_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(
            columns=["normal", "benign", "malignant"],
            fill_value=0,
        )
        .add_suffix("_patient_count")
    )

    total_patient_count = (
        patients.groupby("source_domain")["patient_id"]
        .nunique()
        .rename("total_patient_count")
    )

    diagnoses_per_patient = patients.groupby(["source_domain", "patient_id"])[
        "diagnosis"
    ].nunique()

    multi_diagnosis_patient_count = (
        diagnoses_per_patient.gt(1)
        .groupby(level="source_domain")
        .sum()
        .rename("multi_diagnosis_patient_count")
    )

    summary = diagnosis_counts.join(total_patient_count)
    summary = summary.join(multi_diagnosis_patient_count)

    return summary.reset_index()
