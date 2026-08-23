from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UltrasoundSample:
    image_id: str
    image_path: Path
    mask_path: Path | None

    source_domain: str
    patient_id: str | None

    diagnosis: str | None
    has_lesion: bool

    metadata: dict[str, Any] = field(default_factory=dict)
