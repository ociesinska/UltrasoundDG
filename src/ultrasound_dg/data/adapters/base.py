from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ultrasound_dg.data.sample import UltrasoundSample


class DatasetAdapter(ABC):
    def __init__(self, root: Path):
        self.root = root

    @abstractmethod
    def samples(self) -> list[UltrasoundSample]: ...

    @abstractmethod
    def decode_mask(self, path: Path) -> np.ndarray: ...

    @abstractmethod
    def validate_sample(self, sample: UltrasoundSample) -> None: ...
