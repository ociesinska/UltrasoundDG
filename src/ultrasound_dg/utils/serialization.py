import json
import math
from pathlib import Path
from typing import Any


def make_json_safe(value: Any) -> Any:
    """Replace non-finite floats with JSON-compatible null values."""
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    """Save data as strictly valid, formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(
            make_json_safe(data),
            file,
            indent=2,
            allow_nan=False,
        )
