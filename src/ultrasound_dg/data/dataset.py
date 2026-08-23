from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultrasound_dg.data.sample import UltrasoundSample


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    return image


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not load mask: {path}")

    mask = (mask > 0).astype(np.uint8)

    return mask


class UltrasoundSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[UltrasoundSample],
        transform=None,  # augmentation/preprocessing logic
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = load_image(sample.image_path)

        if sample.mask_path is None:
            mask = np.zeros_like(image, dtype=np.uint8)
        else:
            mask = load_mask(sample.mask_path)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {"image": image, "mask": mask}
