from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config(path: Path, schema: type[ConfigT]) -> ConfigT:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Config must contain a YAML mapping: {path}")

    return schema.model_validate(data)
