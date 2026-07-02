"""Registry and resolver for built-in and custom atlases.

The public entry point is :func:`resolve_atlas`, which turns the user-facing
``atlas`` argument into a validated :class:`pandas.DataFrame`. The argument is
interpreted as:

* a **path to a CSV file** (existing file, or a string ending in ``.csv``) →
  loaded and validated as a custom atlas, or
* a **built-in name** (e.g. ``"Schaefer100-17"``) → loaded from the bundled
  tables, or
* ``None`` → no atlas (returns ``None``).

Names are matched case-insensitively.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Columns every atlas table must provide.
REQUIRED_COLUMNS = ("region", "x", "y", "z")

#: Optional grouping columns used by the report when present.
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
    """Load a built-in atlas by name.

    Parameters
    ----------
    name : str
        A built-in atlas name (case-insensitive), e.g. ``"Schaefer100-17"``.

    Raises
    ------
    AtlasError
        If ``name`` is not a known built-in atlas.
    """
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
    """Resolve the user-facing ``atlas`` argument to a validated DataFrame.

    Parameters
    ----------
    atlas : str or None
        A path to a custom atlas CSV, a built-in atlas name, or ``None``.

    Returns
    -------
    pandas.DataFrame or None
        The atlas table, or ``None`` when ``atlas`` is ``None``.
    """
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
        except Exception as e:  # noqa: BLE001 — surface a clean message
            raise AtlasError(f"Error reading atlas CSV '{path}': {e}") from e
        return _validate(df, path)

    return load_atlas(atlas)
