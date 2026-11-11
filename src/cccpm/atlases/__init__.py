"""Built-in brain atlases (region, x, y, z, ...); pass a name or a custom CSV path."""

from cccpm.atlases.registry import (
    REQUIRED_COLUMNS,
    AtlasError,
    list_atlases,
    load_atlas,
    resolve_atlas,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "AtlasError",
    "list_atlases",
    "load_atlas",
    "resolve_atlas",
]
