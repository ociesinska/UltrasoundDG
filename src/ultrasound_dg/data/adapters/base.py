from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ultrasound_dg.data.sample import UltrasoundSample


class DatasetAdapter(ABC):
    def __init__(self, root: Path):
        """Initialize the adapter with its raw dataset root."""
        self.root = root

    @abstractmethod
    def samples(self) -> list[UltrasoundSample]:
        """Build normalized sample records from the raw dataset."""
        ...

    @abstractmethod
    def decode_mask(self, path: Path) -> np.ndarray:
        """Decode a dataset-specific mask into a binary array."""
        ...

    @abstractmethod
    def validate_sample(self, sample: UltrasoundSample) -> None:
        """Validate dataset-specific image and mask invariants."""
        ...
