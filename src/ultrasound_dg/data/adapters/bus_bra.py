from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.sample import UltrasoundSample
from ultrasound_dg.data.validation import validate_image_mask_pairing


def parse_bus_bra_filename(
    path: Path, option: Literal["usg_image", "mask"]
) -> tuple[str, str]:
    stem = path.stem
    case_part, side = stem.split("-", maxsplit=1)

    if option == "usg_image":
        patient_id = case_part.removeprefix("bus_")

    elif option == "mask":
        patient_id = case_part.removeprefix("mask_")

    else:
        raise ValueError(f"{option} is not accepted. Choose 'usg_image' or 'mask'.")

    laterality_map = {"l": "left", "r": "right", "s": "single"}

    if side not in laterality_map:
        raise ValueError(f"Unexpected BUS-BRA laterality '{side}' in {path.name}")

    return patient_id, laterality_map[side]


def load_bus_bra_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)

    required_columns = {
        "ID",
        "Case",
        "Histology",
        "Pathology",
        "BIRADS",
        "Device",
        "Width",
        "Height",
        "Side",
        "BBOX",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing BUS-BRA metadata columns: {missing_columns}")

    if metadata["ID"].duplicated().any():
        raise ValueError("Duplicate image IDs found in BUS-BRA metadata")

    return metadata.set_index("ID")


class BusBraAdapter(DatasetAdapter):
    def samples(self) -> list[UltrasoundSample]:
        images_path = self.root / "Images"
        masks_path = self.root / "Masks"
        metadata_path = self.root / "bus_data.csv"

        metadata = load_bus_bra_metadata(metadata_path)

        images_by_key: dict[tuple[str, str], Path] = {}

        for image_path in images_path.glob("*.png"):
            patient_id, laterality = parse_bus_bra_filename(
                image_path,
                "usg_image",
            )

            key = (patient_id, laterality)

            if key in images_by_key:
                raise ValueError(f"Duplicate BUS-BRA image key {key}")

            images_by_key[key] = image_path

        masks_by_key: dict[tuple[str, str], Path] = {}

        for mask_path in masks_path.glob("*.png"):
            patient_id, laterality = parse_bus_bra_filename(
                mask_path,
                "mask",
            )

            key = (patient_id, laterality)

            if key in masks_by_key:
                raise ValueError(f"Duplicate BUS-BRA mask key {key}")

            masks_by_key[key] = mask_path

        validate_image_mask_pairing(images_by_key, masks_by_key)

        samples = []

        for key in sorted(images_by_key):
            patient_id, laterality = key

            image_path = images_by_key[key]
            mask_path = masks_by_key[key]

            if image_path.stem not in metadata.index:
                raise ValueError(f"Missing metadata for {image_path.name}")

            row = metadata.loc[image_path.stem]

            diagnosis = row["Pathology"].lower()

            if diagnosis not in {"benign", "malignant"}:
                raise ValueError(
                    f"Unexpected BUS-BRA pathology '{diagnosis}' for {image_path.name}"
                )

            if row["Side"].lower() != laterality:
                raise ValueError(
                    f"Laterality mismatch for {image_path.name}: "
                    f"filename={laterality}, metadata={row['Side']}"
                )

            samples.append(
                UltrasoundSample(
                    image_id=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    source_domain="bus_bra",
                    patient_id=patient_id,
                    diagnosis=diagnosis,
                    has_lesion=True,
                    metadata={
                        "laterality": laterality,
                        "histology": row["Histology"],
                        "birads": row["BIRADS"],
                        "device": row["Device"],
                        "bbox": row["BBOX"],
                    },
                )
            )

        return samples

    def decode_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            if image.mode != "1":
                raise ValueError(
                    f"Expected BUS-BRA mask mode '1', got '{image.mode}' for {path}"
                )

            mask = np.asarray(image)

        return mask.astype(np.uint8)

    def validate_sample(self, sample: UltrasoundSample) -> None:
        mask = self.decode_mask(sample.mask_path)

        if not mask.any():
            raise ValueError(f"Unexpected empty BUS-BRA mask: {sample.mask_path}")

        image = np.asarray(Image.open(sample.image_path))

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image-mask shape mismatch for {sample.image_id}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )
