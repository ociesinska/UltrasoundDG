from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    seed: int
    source_domains: list[str]
    ood_development_domain: list[str]
    final_test_domain: list[str]
    excluded_domains: list[str] = Field(default_factory=list)
    source_validation_fraction: float = Field(gt=0, lt=1)
    splitting: dict[str, SplitConfig]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    train_batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    num_workers: int = Field(default=0, ge=0)
    optimizer: Literal["Adam", "AdamW"] = "AdamW"
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    decision_threshold: float = Field(gt=0, lt=1)
    sampling_strategy: Literal["natural", "domain_balanced"] = "natural"
    checkpoint_selection_tolerance: float = Field(default=0.01, ge=0, le=1)


class TuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_name: str
    mlflow_experiment_name: str
    output_experiment_name: str | None = None
    epochs: int = Field(gt=0)
    learning_rates: list[float] = Field(min_length=1)
    weight_decays: list[float] = Field(min_length=1)
    startup_trials: int = Field(default=4, ge=0)
    warmup_epochs: int = Field(default=10, ge=0)


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    version: Literal["geometric_v1"] = "geometric_v1"

    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    transform_probability: float = Field(default=0.5, ge=0, le=1)
    rotation_limit_degrees: float = Field(default=7.0, ge=0)
    translation_limit: float = Field(default=0.02, ge=0, le=1)
    scale_min: float = Field(default=0.94, gt=0)
    scale_max: float = Field(default=1.06, gt=0)

    @model_validator(mode="after")
    def validate_scale_range(self) -> "AugmentationConfig":
        """Require the minimum scale to be no greater than the maximum."""
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must be less than or equal to scale_max")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    development: DevelopmentConfig
    preprocessing: SegmentationPreprocessingConfig
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    training: TrainingConfig
