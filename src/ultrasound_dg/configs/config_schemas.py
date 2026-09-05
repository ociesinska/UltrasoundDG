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

    train_batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    num_workers: int = Field(default=0, ge=0)
