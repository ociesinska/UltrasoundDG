from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ultrasound_dg.data.adapters.base import DatasetAdapter
from ultrasound_dg.data.sample import UltrasoundSample


def load_breast_usg_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(
        path,
        sheet_name="BrEaST-Lesions-USG clinical dat",
    )

    required_columns = {
        "CaseID",
        "Image_filename",
        "Mask_tumor_filename",
        "Mask_other_filename",
        "Classification",
        "Pixel_size",
        "BIRADS",
        "Diagnosis",
        "Verification",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(f"Missing BrEaST metadata columns: {missing}")

    if metadata["Image_filename"].duplicated().any():
        raise ValueError("Duplicate BrEaST image filenames in metadata")

    return metadata


class BreastUSGAdapter(DatasetAdapter):
    def samples(self) -> list[UltrasoundSample]:
        data_path = self.root
        metadata_path = self.root.with_suffix(".xlsx")

        metadata = load_breast_usg_metadata(metadata_path)

        samples: list[UltrasoundSample] = []

        for _, row in metadata.iterrows():
            image_path = data_path / str(row["Image_filename"])

            if not image_path.exists():
                raise ValueError(f"Missing BrEaST image: {image_path}")

            diagnosis = str(row["Classification"]).strip().lower()

            if diagnosis not in {
                "normal",
                "benign",
                "malignant",
            }:
                raise ValueError(
                    f"Unexpected BrEaST classification "
                    f"'{diagnosis}' for {image_path.name}"
                )

            has_lesion = diagnosis != "normal"

            mask_filename = row["Mask_tumor_filename"]

            if pd.isna(mask_filename):
                if has_lesion:
                    raise ValueError(
                        f"Missing tumor mask metadata for lesion sample "
                        f"{image_path.name}"
                    )

                mask_path = None

            else:
                mask_path = data_path / str(mask_filename).strip()

                if not mask_path.exists():
                    raise ValueError(f"Missing BrEaST tumor mask: {mask_path}")

            other_mask_paths: list[Path] = []

            other_mask_filenames = row["Mask_other_filename"]

            if not pd.isna(other_mask_filenames):
                for filename in str(other_mask_filenames).split("&"):
                    filename = filename.strip()

                    other_path = data_path / filename

                    if not other_path.exists():
                        raise ValueError(f"Missing BrEaST other mask: {other_path}")

                    other_mask_paths.append(other_path)

            samples.append(
                UltrasoundSample(
                    image_id=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    source_domain="breast_usg",
                    patient_id=str(row["CaseID"]),
                    diagnosis=diagnosis,
                    has_lesion=has_lesion,
                    metadata={
                        "pixel_size": row["Pixel_size"],
                        "birads": row["BIRADS"],
                        "histologic_diagnosis": row["Diagnosis"],
                        "verification": row["Verification"],
                        "other_mask_paths": tuple(other_mask_paths),
                    },
                )
            )

        return samples

    def decode_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            if image.mode != "RGBA":
                raise ValueError(
                    f"Expected BrEaST mask mode 'RGBA', got '{image.mode}' for {path}"
                )

            mask = np.asarray(image)

        return (mask[:, :, 3] > 0).astype(np.uint8)

    def validate_sample(self, sample: UltrasoundSample) -> None:
        with Image.open(sample.image_path) as image:
            image_shape = np.asarray(image).shape[:2]

        if sample.mask_path is None:
            if sample.has_lesion:
                raise ValueError(f"Lesion sample {sample.image_id} has no tumor mask")

            return

        mask = self.decode_mask(sample.mask_path)

        if image_shape != mask.shape:
            raise ValueError(
                f"Image-mask shape mismatch for {sample.image_id}: "
                f"image={image_shape}, mask={mask.shape}"
            )

        if sample.has_lesion and not mask.any():
            raise ValueError(f"Expected non-empty tumor mask for {sample.image_id}")

        if not sample.has_lesion and mask.any():
            raise ValueError(
                f"Normal sample {sample.image_id} has non-empty tumor mask"
            )
