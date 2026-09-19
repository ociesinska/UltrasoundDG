from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SegmentationPreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "v1"
    target_size: int = 256


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["patient", "sample"]
    stratify_by: str | None = None


class DevelopmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    seed: int
    source_domains: list[str]
    ood_development_domain: list[str]
    final_test_domain: list[str]
    source_validation_fraction: float = Field(gt=0, lt=1)
    splitting: dict[str, SplitConfig]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    mlflow_experiment_name: str
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    train_batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    num_workers: int = Field(default=0, ge=0)
    optimizer: Literal["Adam", "AdamW"] = "AdamW"
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    decision_threshold: float = Field(gt=0, lt=1)


class TuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_name: str
    mlflow_experiment_name: str
    epochs: int = Field(gt=0)
    learning_rates: list[float] = Field(min_length=1)
    weight_decays: list[float] = Field(min_length=1)
    startup_trials: int = Field(default=4, ge=0)
    warmup_epochs: int = Field(default=10, ge=0)
