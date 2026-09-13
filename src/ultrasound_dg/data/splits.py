import math

import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)

from ultrasound_dg.configs.config_schemas import DevelopmentConfig


def split_by_patient(
    df: pd.DataFrame,
    validation_fraction: float,
    stratify_by: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df["patient_id"].isna().any():
        raise ValueError("Patient-level split requires a patient_id for every sample.")

    if stratify_by is None:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=seed,
        )
        train_idx, val_idx = next(
            splitter.split(
                X=df,
                groups=df["patient_id"],
            )
        )
    else:
        _require_column(df, stratify_by)
        n_splits = _fold_count(validation_fraction)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        train_idx, val_idx = next(
            splitter.split(
                X=df,
                y=df[stratify_by],
                groups=df["patient_id"],
            )
        )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    if not train_patients.isdisjoint(val_patients):
        raise RuntimeError("Patient leakage detected between train and validation.")

    return train_df, val_df


def split_by_sample(
    df: pd.DataFrame,
    validation_fraction: float,
    stratify_by: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    if stratify_by is not None:
        _require_column(df, stratify_by)
        stratify = df[stratify_by]

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=validation_fraction,
        random_state=seed,
        stratify=stratify,
    )

    train_df = df.loc[train_idx].copy()
    val_df = df.loc[val_idx].copy()

    return train_df, val_df


def create_development_protocol(
    manifest: pd.DataFrame,
    config: DevelopmentConfig,
) -> dict[str, pd.DataFrame]:
    _validate_protocol_config(manifest, config)

    train_parts: list[pd.DataFrame] = []
    source_val_parts: list[pd.DataFrame] = []

    for source_domain in config.source_domains:
        domain_manifest = manifest.loc[
            manifest["source_domain"] == source_domain
        ].copy()
        split_config = config.splitting[source_domain]

        if split_config.strategy == "patient":
            train_part, val_part = split_by_patient(
                domain_manifest,
                validation_fraction=config.source_validation_fraction,
                stratify_by=split_config.stratify_by,
                seed=config.seed,
            )
        else:
            train_part, val_part = split_by_sample(
                domain_manifest,
                validation_fraction=config.source_validation_fraction,
                stratify_by=split_config.stratify_by,
                seed=config.seed,
            )

        train_parts.append(train_part)
        source_val_parts.append(val_part)

    protocol = {
        "train": pd.concat(train_parts, ignore_index=True),
        "source_val": pd.concat(source_val_parts, ignore_index=True),
        "ood_dev": manifest.loc[
            manifest["source_domain"].isin(config.ood_development_domain)
        ]
        .copy()
        .reset_index(drop=True),
        "final_test": manifest.loc[
            manifest["source_domain"].isin(config.final_test_domain)
        ]
        .copy()
        .reset_index(drop=True),
    }

    _validate_protocol_partitions(manifest, protocol)

    return protocol


def _fold_count(validation_fraction: float) -> int:
    inverse_fraction = 1.0 / validation_fraction
    n_splits = round(inverse_fraction)

    if n_splits < 2 or not math.isclose(
        inverse_fraction,
        n_splits,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "A stratified patient split requires source_validation_fraction "
            "to be the reciprocal of an integer greater than one, for example "
            "0.2, 0.25, or 0.1."
        )

    return n_splits


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Split column is missing from the manifest: {column}")


def _validate_protocol_config(
    manifest: pd.DataFrame,
    config: DevelopmentConfig,
) -> None:
    _require_column(manifest, "source_domain")
    _require_column(manifest, "sample_id")

    if not config.source_domains:
        raise ValueError("At least one source domain must be configured.")

    source_domains = set(config.source_domains)
    ood_domains = set(config.ood_development_domain)
    final_domains = set(config.final_test_domain)

    if (
        source_domains & ood_domains
        or source_domains & final_domains
        or ood_domains & final_domains
    ):
        raise ValueError(
            "Source, OOD development, and final test domains must be disjoint."
        )

    split_domains = set(config.splitting)
    if split_domains != source_domains:
        missing = sorted(source_domains - split_domains)
        unexpected = sorted(split_domains - source_domains)
        raise ValueError(
            "Splitting configuration must match source_domains exactly. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )

    configured_domains = source_domains | ood_domains | final_domains
    manifest_domains = set(manifest["source_domain"].dropna())

    missing_from_manifest = configured_domains - manifest_domains
    unassigned = manifest_domains - configured_domains
    if missing_from_manifest or unassigned:
        raise ValueError(
            "Configured domains and manifest domains do not match. "
            f"Missing from manifest: {sorted(missing_from_manifest)}; "
            f"unassigned manifest domains: {sorted(unassigned)}."
        )


def _validate_protocol_partitions(
    manifest: pd.DataFrame,
    protocol: dict[str, pd.DataFrame],
) -> None:
    original_ids = manifest["sample_id"]
    partitioned_ids = pd.concat(
        [partition["sample_id"] for partition in protocol.values()],
        ignore_index=True,
    )

    if original_ids.duplicated().any():
        raise ValueError("Manifest sample_id values must be unique before splitting.")

    if partitioned_ids.duplicated().any():
        raise RuntimeError("A sample occurs in more than one protocol partition.")

    if set(partitioned_ids) != set(original_ids):
        raise RuntimeError("Protocol partitions do not cover the manifest exactly.")
