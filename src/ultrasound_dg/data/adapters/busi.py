from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.sample import UltrasoundSample
from ultrasound_dg.data.validation import validate_image_mask_pairing


def parse_busi_filename(
    path: Path,
    option: Literal["usg_image", "mask"],
) -> tuple[str, str]:
    stem = path.stem

    if option == "mask":
        stem = stem.removesuffix("_mask")
    elif option != "usg_image":
        raise ValueError(f"{option} is not accepted. Choose 'usg_image' or 'mask'.")

    diagnosis, token, case_id = stem.split("_", maxsplit=2)

    if token != "id":
        raise ValueError(f"Unexpected BUSI filename format: {path.name}")

    return case_id, diagnosis


class BusiAdapter(DatasetAdapter):
    def samples(self) -> list[UltrasoundSample]:

        images_path = self.root / "images"
        masks_path = self.root / "masks"

        images_by_key: dict[tuple[str, str], Path] = {}

        for image_path in images_path.glob("*.png"):
            case_id, diagnosis = parse_busi_filename(image_path, "usg_image")

            key = (case_id, diagnosis)

            if key in images_by_key:
                raise ValueError(f"Duplicate BUSI image key {key}")

            images_by_key[key] = image_path

        masks_by_key: dict[tuple[str, str], Path] = {}

        for mask_path in masks_path.glob("*.png"):
            case_id, diagnosis = parse_busi_filename(mask_path, "mask")

            key = (case_id, diagnosis)

            if key in masks_by_key:
                raise ValueError(f"Duplicate BUSI mask key {key}")

            masks_by_key[key] = mask_path

        validate_image_mask_pairing(images_by_key, masks_by_key)

        samples = []

        for key in sorted(images_by_key):
            case_id, diagnosis = key

            image_path = images_by_key[key]
            mask_path = masks_by_key[key]

            samples.append(
                UltrasoundSample(
                    image_id=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    source_domain="busi",
                    patient_id=None,
                    diagnosis=diagnosis,
                    has_lesion=diagnosis != "normal",
                    metadata={},
                )
            )

        return samples

    def validate_sample(self, sample: UltrasoundSample) -> None:
        mask = self.decode_mask(sample.mask_path)
        image = np.asarray(Image.open(sample.image_path))

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image-mask shape mismatch for {sample.image_id}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )
        if sample.has_lesion and not mask.any():
            raise ValueError(f"Expected non-empty mask for {sample.image_id}")

        if not sample.has_lesion and mask.any():
            raise ValueError(f"Expected empty mask for normal sample {sample.image_id}")

    def decode_mask(self, path: Path) -> np.ndarray:

        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"Expected BUSI mask mode 'L', got '{image.mode}'")

            mask = np.asarray(image)

            mask = (mask >= 254).astype(np.uint8)

        return mask
