from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.sample import UltrasoundSample
from ultrasound_dg.data.validation import validate_image_mask_pairing

EXCLUDED_PATIENTS = {
    "HESN": (
        "Image dimensions do not match annotation dimensions for all scans of this patient."
    )
}


def parse_bus_uclm_filename(path: Path) -> tuple[str, str]:
    patient_id, scan_id = path.stem.split("_", maxsplit=1)

    return patient_id, scan_id


def load_bus_uclm_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep=";")

    required_columns = {
        "Image",
        "Resolution",
        "Label",
        "Doppler",
        "Marks",
        "Combined",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing BUS-UCLM metadata columns: {missing_columns}")

    if metadata["Image"].duplicated().any():
        raise ValueError("Duplicate image names found in BUS-UCLM metadata")

    return metadata.set_index("Image")


class BusUclmAdapter(DatasetAdapter):
    def samples(self) -> list[UltrasoundSample]:

        images_path = self.root / "images"
        masks_path = self.root / "masks"
        metadata_path = self.root / "INFO.csv"

        metadata = load_bus_uclm_metadata(metadata_path)

        images_by_key: dict[tuple[str, str], Path] = {}

        for image_path in images_path.glob("*.png"):
            patient_id, scan_id = parse_bus_uclm_filename(image_path)

            key = (patient_id, scan_id)

            if key in images_by_key:
                raise ValueError(f"Duplicate BUS UCLM image key {key}")

            images_by_key[key] = image_path

        masks_by_key: dict[tuple[str, str], Path] = {}

        for mask_path in masks_path.glob("*.png"):
            patient_id, scan_id = parse_bus_uclm_filename(mask_path)

            key = (patient_id, scan_id)

            if key in masks_by_key:
                raise ValueError(f"Duplicate BUS UCLM mask key {key}")

            masks_by_key[key] = mask_path

        validate_image_mask_pairing(images_by_key, masks_by_key)

        samples = []

        for key in sorted(images_by_key):
            patient_id, scan_id = key

            if patient_id in EXCLUDED_PATIENTS:
                continue

            image_path = images_by_key[key]
            mask_path = masks_by_key[key]

            if image_path.name not in metadata.index:
                raise ValueError(f"Missing metadata for {image_path.name}")

            row = metadata.loc[image_path.name]
            diagnosis = row["Label"].lower()
            has_lesion = diagnosis != "normal"

            samples.append(
                UltrasoundSample(
                    image_id=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    source_domain="bus_uclm",
                    patient_id=patient_id,
                    diagnosis=diagnosis,
                    has_lesion=has_lesion,
                    metadata={
                        "resolution": row["Resolution"],
                        "doppler": row["Doppler"] == "Yes",
                        "marks": row["Marks"] == "Yes",
                        "combined": row["Combined"] == "Yes",
                    },
                )
            )
        return samples

    def decode_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(
                    f"Expected BUS-UCLM mask mode 'RGB', got '{image.mode}' for {path}"
                )

            mask = np.asarray(image)

        benign = np.all(mask == [0, 255, 0], axis=-1)
        malignant = np.all(mask == [255, 0, 0], axis=-1)

        return (benign | malignant).astype(np.uint8)

    def validate_sample(self, sample: UltrasoundSample) -> None:
        mask = self.decode_mask(sample.mask_path)

        with Image.open(sample.image_path) as image:
            image_shape = np.asarray(image).shape[:2]

        if image_shape != mask.shape:
            raise ValueError(
                f"Image-mask shape mismatch for {sample.image_id}: "
                f"image={image_shape[:2]}, mask={mask.shape}"
            )
        if sample.has_lesion and not mask.any():
            raise ValueError(f"Expected non-empty mask for {sample.image_id}")

        if not sample.has_lesion and mask.any():
            raise ValueError(f"Expected empty mask for normal sample {sample.image_id}")
