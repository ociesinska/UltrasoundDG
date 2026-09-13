from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.image_io import load_rgb_image
from ultrasound_dg.data.preprocessing import SegmentationPreprocessor


class UltrasoundSegmentationDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        project_root: Path,
        adapters: dict[str, DatasetAdapter],
        preprocessor: SegmentationPreprocessor,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.project_root = project_root
        self.adapters = adapters
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]

        image_path = self.project_root / row["image_path"]
        image = load_rgb_image(image_path)

        if pd.isna(row["mask_path"]):
            # Some normal cases do not provide a mask file.
            # Empty mask represents background-only ground truth.
            mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8,
            )

        else:
            mask_path = self.project_root / row["mask_path"]
            adapter = self.adapters[row["source_domain"]]
            mask = adapter.decode_mask(mask_path)

        processed = self.preprocessor(image=image, mask=mask)
        image = processed["image"]
        mask = processed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "sample_id": row["sample_id"],
            "source_domain": row["source_domain"],
            "diagnosis": row["diagnosis"],
            "has_lesion": bool(row["has_lesion"]),
        }
