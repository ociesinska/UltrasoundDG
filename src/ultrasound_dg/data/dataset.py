from pathlib import Path

import albumentations as A
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
        augmenter: A.Compose | None = None,
    ):
        """Initialize dataset access from a manifest and domain adapters."""
        self.manifest = manifest.reset_index(drop=True)
        self.project_root = project_root
        self.adapters = adapters
        self.preprocessor = preprocessor
        self.augmenter = augmenter

    def __len__(self) -> int:
        """Return the number of manifest samples."""
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        """Load and preprocess one image-mask pair with sample metadata."""
        applied_transforms = None
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

        if self.augmenter is not None:
            augmented = self.augmenter(
                image=image,
                mask=mask,
            )

            image = augmented["image"]
            mask = augmented["mask"]

            applied_transforms = augmented.get("applied_transforms")

        processed = self.preprocessor(image=image, mask=mask)
        image = processed["image"]
        mask = processed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0)

        patient_id = row.get("patient_id", pd.NA)
        patient_id = "" if pd.isna(patient_id) else str(patient_id)

        sample = {
            "image": image,
            "mask": mask,
            "sample_id": row["sample_id"],
            "source_domain": row["source_domain"],
            "patient_id": patient_id,
            "diagnosis": row["diagnosis"],
            "has_lesion": bool(row["has_lesion"]),
        }

        if applied_transforms is not None:
            sample["applied_transforms"] = applied_transforms

        return sample
