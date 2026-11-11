"""Registry and resolver for built-in and custom (CSV) atlases."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

REQUIRED_COLUMNS = ("region", "x", "y", "z")
OPTIONAL_COLUMNS = ("network", "hemisphere", "structure")


class AtlasError(ValueError):
    """Raised when an atlas cannot be resolved, loaded, or validated."""


@lru_cache(maxsize=1)
def _bundled_index() -> dict[str, Path]:
    """Map lowercased built-in atlas name -> bundled CSV path."""
    if not DATA_DIR.is_dir():
        return {}
    return {
        p.stem.lower(): p
        for p in sorted(DATA_DIR.glob("*.csv"))
    }


def list_atlases() -> list[str]:
    """Return the names of the built-in atlases, sorted."""
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))


def _validate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Check required columns and coordinate types; return the frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AtlasError(
            f"Atlas '{source}' is missing required column(s): "
            f"{', '.join(missing)}. Required columns are {list(REQUIRED_COLUMNS)}."
        )
    for c in ("x", "y", "z"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise AtlasError(
                f"Atlas '{source}' column '{c}' must be numeric (MNI coordinate)."
            )
    return df


def _looks_like_path(value: str) -> bool:
    """True if the string should be treated as a CSV path rather than a name."""
    return value.lower().endswith(".csv") or os.path.isfile(value)


def load_atlas(name: str) -> pd.DataFrame:
    """Load a built-in atlas by name (case-insensitive)."""
    index = _bundled_index()
    path = index.get(name.lower())
    if path is None:
        available = ", ".join(list_atlases()) or "(none bundled)"
        raise AtlasError(
            f"Unknown built-in atlas '{name}'. Available atlases: {available}. "
            f"To use a custom atlas, pass a path to a CSV file with columns "
            f"{list(REQUIRED_COLUMNS)}."
        )
    return _validate(pd.read_csv(path), name)


def resolve_atlas(atlas: Optional[str]) -> Optional[pd.DataFrame]:
    """Resolve ``atlas``: a CSV path, a built-in atlas name, or ``None``."""
    if atlas is None:
        return None
    if not isinstance(atlas, str):
        raise AtlasError(
            f"`atlas` must be a string (built-in name or path to a CSV) or None, "
            f"got {type(atlas).__name__}."
        )

    if _looks_like_path(atlas):
        path = os.path.abspath(atlas)
        if not os.path.isfile(path):
            raise AtlasError(f"Atlas CSV file does not exist: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            raise AtlasError(f"Error reading atlas CSV '{path}': {e}") from e
        return _validate(df, path)

    return load_atlas(atlas)
