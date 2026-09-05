import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def split_by_patient(
    df: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    train_idx, val_idx = next(
        splitter.split(
            X=df,
            y=df[
                "diagnosis"
            ],  # to keep the proportions of benign/malignant in both splits
            groups=df["patient_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    assert set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"]))

    return train_df, val_df


def split_sample(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_idx, val_idx = train_test_split(
        df.index, test_size=0.2, random_state=seed, stratify=df["diagnosis"]
    )

    train_df = df.loc[train_idx].copy()
    val_df = df.loc[val_idx].copy()

    return train_df, val_df


def create_development_protocol(
    manifest: pd.DataFrame,
    seed: int = 42,
):

    bus_bra = manifest[manifest["source_domain"] == "bus_bra"]
    busi = manifest[manifest["source_domain"] == "busi"]
    bus_uclm = manifest[manifest["source_domain"] == "bus_uclm"]
    breast_usg = manifest[manifest["source_domain"] == "breast_usg"]

    bus_bra_train, bus_bra_val = split_by_patient(
        bus_bra,
        seed=seed,
    )

    busi_train, busi_val = split_sample(
        busi,
        seed=seed,
    )

    train = pd.concat([bus_bra_train, busi_train], ignore_index=True)
    source_val = pd.concat([bus_bra_val, busi_val], ignore_index=True)

    ood_dev = bus_uclm.copy()
    final_test = breast_usg.copy()

    assert len(train) + len(source_val) + len(ood_dev) + len(final_test) == len(
        manifest
    )
    assert set(bus_bra_train["patient_id"]).isdisjoint(set(bus_bra_val["patient_id"]))

    protocol = {
        "train": train,
        "source_val": source_val,
        "ood_dev": ood_dev,
        "final_test": final_test,
    }

    return protocol
