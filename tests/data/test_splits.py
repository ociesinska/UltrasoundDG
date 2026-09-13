import pandas as pd

from ultrasound_dg.configs.config_schemas import DevelopmentConfig, SplitConfig
from ultrasound_dg.data.splits import create_development_protocol


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for index in range(16):
        rows.append(
            {
                "sample_id": f"patient_source_{index}",
                "source_domain": "patient_source",
                "patient_id": f"patient_{index}",
                "diagnosis": "benign" if index % 2 == 0 else "malignant",
            }
        )

    for index in range(20):
        rows.append(
            {
                "sample_id": f"sample_source_{index}",
                "source_domain": "sample_source",
                "patient_id": pd.NA,
                "diagnosis": "normal" if index % 2 == 0 else "benign",
            }
        )

    for domain, count in (("external_dev", 3), ("locked_test", 2)):
        for index in range(count):
            rows.append(
                {
                    "sample_id": f"{domain}_{index}",
                    "source_domain": domain,
                    "patient_id": f"{domain}_patient_{index}",
                    "diagnosis": "benign",
                }
            )

    return pd.DataFrame(rows)


def _config() -> DevelopmentConfig:
    return DevelopmentConfig(
        version="test",
        seed=7,
        source_domains=["patient_source", "sample_source"],
        ood_development_domain=["external_dev"],
        final_test_domain=["locked_test"],
        source_validation_fraction=0.25,
        splitting={
            "patient_source": SplitConfig(
                strategy="patient",
                stratify_by="diagnosis",
            ),
            "sample_source": SplitConfig(
                strategy="sample",
                stratify_by="diagnosis",
            ),
        },
    )


def test_development_protocol_uses_configured_domains_and_strategies() -> None:
    manifest = _manifest()

    protocol = create_development_protocol(manifest, _config())

    assert set(protocol) == {"train", "source_val", "ood_dev", "final_test"}
    assert set(protocol["train"]["source_domain"]) == {
        "patient_source",
        "sample_source",
    }
    assert set(protocol["source_val"]["source_domain"]) == {
        "patient_source",
        "sample_source",
    }
    assert set(protocol["ood_dev"]["source_domain"]) == {"external_dev"}
    assert set(protocol["final_test"]["source_domain"]) == {"locked_test"}

    patient_train = protocol["train"].query("source_domain == 'patient_source'")
    patient_val = protocol["source_val"].query("source_domain == 'patient_source'")
    assert set(patient_train["patient_id"]).isdisjoint(patient_val["patient_id"])
    assert len(patient_val) == 4

    sample_val = protocol["source_val"].query("source_domain == 'sample_source'")
    assert len(sample_val) == 5

    partitioned_ids = pd.concat(
        [partition["sample_id"] for partition in protocol.values()],
        ignore_index=True,
    )
    assert set(partitioned_ids) == set(manifest["sample_id"])
    assert not partitioned_ids.duplicated().any()


def test_development_protocol_is_reproducible() -> None:
    manifest = _manifest()
    config = _config()

    first = create_development_protocol(manifest, config)
    second = create_development_protocol(manifest, config)

    for partition_name in first:
        assert (
            first[partition_name]["sample_id"].tolist()
            == second[partition_name]["sample_id"].tolist()
        )
