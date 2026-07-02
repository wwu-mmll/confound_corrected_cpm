"""Built-in brain atlases with region names and MNI coordinates.

CCCPM ships vetted coordinate tables for common parcellations so users can
select an atlas by name (``atlas="Schaefer100-17"``) instead of hand-building a
CSV. Custom atlases are still supported by passing a path to a CSV file.

The bundled tables live in ``data/`` and are regenerated from authentic public
sources by ``scripts/generate_atlas_data.py`` (see ``data/SOURCES.md`` for
provenance). Every table has columns ``region, x, y, z`` and, where available,
``network``, plus anatomical fallback columns ``hemisphere`` and ``structure``.
"""

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
