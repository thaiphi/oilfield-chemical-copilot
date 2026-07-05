from __future__ import annotations

from pathlib import Path
from typing import Literal

DataMode = Literal["sample", "private"]

DATA_ROOTS: dict[DataMode, Path] = {
    "sample": Path("data/sample"),
    "private": Path("data/private"),
}


def data_root_for_mode(mode: DataMode) -> Path:
    """Return the configured data root for a supported ingestion mode."""
    if mode not in DATA_ROOTS:
        raise ValueError(f"Unsupported data mode: {mode}")
    return DATA_ROOTS[mode]
