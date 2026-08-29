from pathlib import Path

import pandas as pd

from ultrasound_dg.data.sample import UltrasoundSample

REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "image_id",
    "source_domain",
    "patient_id",
    "image_path",
    "mask_path",
    "diagnosis",
    "has_lesion",
}

MANIFEST_STRING_COLUMNS = {
    "sample_id": "string",
    "image_id": "string",
    "source_domain": "string",
    "patient_id": "string",
    "image_path": "string",
    "mask_path": "string",
    "diagnosis": "string",
}

VALID_DIAGNOSES = {"normal", "benign", "malignant"}


def samples_to_manifest(
    samples: list[UltrasoundSample],
    project_root: Path,
) -> pd.DataFrame:
    rows = []

    for sample in samples:
        row = {
            "sample_id": f"{sample.source_domain}_{sample.image_id}",
            "image_id": f"{sample.image_id}",
            "source_domain": sample.source_domain,
            "patient_id": sample.patient_id,
            "image_path": sample.image_path.relative_to(project_root),
            "mask_path": (
                sample.mask_path.relative_to(project_root)
                if sample.mask_path is not None
                else None
            ),
            "diagnosis": sample.diagnosis,
            "has_lesion": sample.has_lesion,
        }

        metadata = dict(sample.metadata)

        other_mask_paths = metadata.get("other_mask_paths")

        if other_mask_paths:
            metadata["other_mask_paths"] = "|".join(
                str(path.relative_to(project_root)) for path in other_mask_paths
            )

        row.update(metadata)

        rows.append(row)

    return pd.DataFrame(rows)


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            **MANIFEST_STRING_COLUMNS,
            "has_lesion": "boolean",
        },
    )


def validate_manifest(
    manifest: pd.DataFrame,
    project_root: Path,
) -> None:
    errors: list[str] = []

    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required manifest columns: {sorted(missing_columns)}"
        )

    if manifest.empty:
        errors.append("manifest contains no samples")

    required_non_null = {
        "sample_id",
        "image_id",
        "source_domain",
        "image_path",
        "diagnosis",
        "has_lesion",
    }

    for column in required_non_null:
        if manifest[column].isna().any():
            errors.append(f"column '{column}' contains missing values")

    if manifest["sample_id"].duplicated().any():
        errors.append("sample_id values are not unique")

    if manifest["image_path"].duplicated().any():
        errors.append("image_path values are not unique")

    invalid_diagnoses = set(manifest["diagnosis"].dropna()) - VALID_DIAGNOSES

    if invalid_diagnoses:
        errors.append(f"unexpected diagnoses: {sorted(invalid_diagnoses)}")

    inconsistent = manifest["has_lesion"] != (manifest["diagnosis"] != "normal")

    if inconsistent.any():
        errors.append("diagnosis and has_lesion are inconsistent")

    missing_mask = manifest["mask_path"].isna()

    lesion_without_mask = manifest["has_lesion"] & missing_mask

    if lesion_without_mask.any():
        errors.append("some lesion samples have no mask")

    for column in ("image_path", "mask_path"):
        for value in manifest[column].dropna():
            path = Path(value)

            if path.is_absolute():
                errors.append(f"{column} contains absolute path: {path}")
                continue

            if not (project_root / path).is_file():
                errors.append(f"{column} references missing file: {path}")

    if errors:
        raise ValueError("Invalid manifest:\n- " + "\n- ".join(errors))
