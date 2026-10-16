from typing import Annotated, Literal

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


class GeometricAugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["geometric_v1"]
    enabled: Literal[True] = True

    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    transform_probability: float = Field(default=0.5, ge=0, le=1)
    rotation_limit_degrees: float = Field(default=7.0, ge=0)
    translation_limit: float = Field(default=0.02, ge=0, le=1)
    scale_min: float = Field(default=0.94, gt=0)
    scale_max: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_scale_range(self) -> "GeometricAugmentationConfig":
        """Require the minimum scale to be no greater than the maximum."""
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must be less than or equal to scale_max")
        return self


class NoAugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["none"] = "none"
    enabled: Literal[False] = False


class IntensityAugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["intensity_v1"]
    enabled: Literal[True] = True

    transform_probability: float = Field(default=0.5, ge=0, le=1)
    brightness_limit: float = Field(default=0.10, ge=0, le=1)
    contrast_limit: float = Field(default=0.10, ge=0, le=1)
    gamma_min: float = Field(default=0.90, gt=0)
    gamma_max: float = Field(default=1.10, gt=0)
    noise_std_max: float = Field(default=0.02, ge=0, le=1)
    blur_kernel_size: int = Field(default=3, gt=0)
    blur_sigma_max: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_intensity_ranges(
        self,
    ) -> "IntensityAugmentationConfig":
        """Validate intensity augmentation ranges."""
        if self.gamma_min > self.gamma_max:
            raise ValueError("gamma_min must be less than or equal to gamma_max")

        if self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")

        return self


AugmentationConfig = Annotated[
    NoAugmentationConfig | GeometricAugmentationConfig | IntensityAugmentationConfig,
    Field(discriminator="version"),
]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    development: DevelopmentConfig
    preprocessing: SegmentationPreprocessingConfig
    augmentation: AugmentationConfig = Field(default_factory=NoAugmentationConfig)
    training: TrainingConfig

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_disabled_augmentation(cls, data):
        """Map legacy disabled geometric configs to the explicit no-op variant."""
        if not isinstance(data, dict):
            return data

        augmentation = data.get("augmentation")

        if isinstance(augmentation, dict) and augmentation.get("enabled") is False:
            return {
                **data,
                "augmentation": {
                    "version": "none",
                    "enabled": False,
                },
            }

        return data
