from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.dataset import UltrasoundSegmentationDataset
from ultrasound_dg.data.preprocessing import (
    SegmentationPreprocessingConfig,
    SegmentationPreprocessor,
)
from ultrasound_dg.data.sample import UltrasoundSample


class StubAdapter(DatasetAdapter):
    def samples(self) -> list[UltrasoundSample]:
        return []

    def decode_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return (np.asarray(image) > 0).astype(np.uint8)

    def validate_sample(self, sample: UltrasoundSample) -> None:
        pass


def test_dataset_returns_model_ready_tensors(tmp_path: Path) -> None:
    image = np.full((4, 6, 3), fill_value=128, dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:4] = 255

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "test_lesion",
                "source_domain": "test",
                "image_path": image_path.name,
                "mask_path": mask_path.name,
                "diagnosis": "benign",
                "has_lesion": True,
            },
            {
                "sample_id": "test_normal",
                "source_domain": "test",
                "image_path": image_path.name,
                "mask_path": None,
                "diagnosis": "normal",
                "has_lesion": False,
            },
        ]
    )
    preprocessor = SegmentationPreprocessor(
        SegmentationPreprocessingConfig(target_size=8)
    )
    dataset = UltrasoundSegmentationDataset(
        manifest=manifest,
        project_root=tmp_path,
        adapters={"test": StubAdapter(tmp_path)},
        preprocessor=preprocessor,
    )

    lesion_sample = dataset[0]
    normal_sample = dataset[1]

    assert lesion_sample["image"].shape == (3, 8, 8)
    assert lesion_sample["mask"].shape == (1, 8, 8)
    assert lesion_sample["image"].dtype == torch.float32
    assert lesion_sample["mask"].dtype == torch.float32
    assert torch.isfinite(lesion_sample["image"]).all()
    assert set(torch.unique(lesion_sample["mask"]).tolist()) <= {0.0, 1.0}
    assert lesion_sample["mask"].any()
    assert not normal_sample["mask"].any()
    assert lesion_sample["sample_id"] == "test_lesion"
    assert lesion_sample["source_domain"] == "test"
    assert lesion_sample["diagnosis"] == "benign"
    assert lesion_sample["has_lesion"] is True
    assert normal_sample["diagnosis"] == "normal"
    assert normal_sample["has_lesion"] is False
