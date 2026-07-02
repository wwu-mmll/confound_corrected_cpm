"""Tests for the built-in atlas registry and the ``atlas`` parameter."""

import warnings

import numpy as np
import pandas as pd
import pytest

from cccpm.atlases import (
    REQUIRED_COLUMNS,
    AtlasError,
    list_atlases,
    load_atlas,
    resolve_atlas,
)
from cccpm.cpm_analysis import CPMAnalysis


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_list_atlases_nonempty():
    atlases = list_atlases()
    assert atlases, "expected at least one bundled atlas"
    assert "Schaefer100-17" in atlases


def test_bundled_atlases_have_required_schema():
    for name in list_atlases():
        df = load_atlas(name)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"{name} missing {col}"
        assert len(df) > 0
        for col in ("x", "y", "z"):
            assert pd.api.types.is_numeric_dtype(df[col]), f"{name}.{col} not numeric"


def test_schaefer_node_counts_and_networks():
    df = load_atlas("Schaefer100-17")
    assert len(df) == 100
    assert "network" in df.columns
    # 17-network parcellation should expose more networks than the 7-network one.
    assert df["network"].nunique() > load_atlas("Schaefer100-7")["network"].nunique()


def test_expected_atlases_are_bundled():
    # A representative set spanning coordinate-defined, volumetric, and
    # surface-derived (volumetric-MNI) parcellations.
    expected = {
        "Schaefer100-7", "Schaefer400-17", "Power264", "Dosenbach160",
        "Seitzman300", "Destrieux148", "HarvardOxfordCortical",
        "AAL116", "Glasser360",
    }
    assert expected <= set(list_atlases())


def test_surface_derived_atlases_have_node_counts_and_hemispheres():
    destrieux = load_atlas("Destrieux148")
    assert len(destrieux) == 148
    # Destrieux is split evenly across hemispheres.
    assert set(destrieux["hemisphere"]) >= {"L", "R"}

    ho = load_atlas("HarvardOxfordCortical")
    assert len(ho) == 96
    assert set(ho["hemisphere"]) >= {"L", "R"}


def test_volumetric_atlases_node_counts():
    assert len(load_atlas("AAL116")) == 116
    assert len(load_atlas("Glasser360")) == 360
    # AAL spans cortex, subcortex and cerebellum; Glasser is cortical only.
    assert set(load_atlas("AAL116")["structure"]) >= {"cortical", "subcortical", "cerebellum"}
    assert set(load_atlas("Glasser360")["hemisphere"]) == {"L", "R"}


def test_load_atlas_is_case_insensitive():
    assert len(load_atlas("schaefer100-17")) == len(load_atlas("Schaefer100-17"))


def test_load_unknown_atlas_raises():
    with pytest.raises(AtlasError, match="Unknown built-in atlas"):
        load_atlas("NotARealAtlas")


# ---------------------------------------------------------------------------
# resolve_atlas: path vs name vs None
# ---------------------------------------------------------------------------

def test_resolve_none_returns_none():
    assert resolve_atlas(None) is None


def test_resolve_builtin_name():
    df = resolve_atlas("Power264")
    assert len(df) == 264


def test_resolve_custom_csv_path(tmp_path):
    csv = tmp_path / "custom.csv"
    pd.DataFrame(
        {"region": ["a", "b"], "x": [1, 2], "y": [3, 4], "z": [5, 6]}
    ).to_csv(csv, index=False)
    df = resolve_atlas(str(csv))
    assert list(df["region"]) == ["a", "b"]


def test_resolve_missing_csv_path_raises(tmp_path):
    with pytest.raises(AtlasError, match="does not exist"):
        resolve_atlas(str(tmp_path / "nope.csv"))


def test_resolve_custom_csv_missing_columns_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"region": ["a"], "x": [1]}).to_csv(csv, index=False)
    with pytest.raises(AtlasError, match="missing required column"):
        resolve_atlas(str(csv))


# ---------------------------------------------------------------------------
# CPMAnalysis integration
# ---------------------------------------------------------------------------

def test_cpm_atlas_by_name_is_saved(tmp_path):
    cpm = CPMAnalysis(results_directory=str(tmp_path / "res"), atlas="Schaefer100-17")
    assert cpm.atlas is not None and len(cpm.atlas) == 100
    saved = pd.read_csv(cpm.atlas_labels)
    assert len(saved) == 100


def test_atlas_labels_is_deprecated(tmp_path):
    csv = tmp_path / "custom.csv"
    pd.DataFrame(
        {"region": ["a", "b"], "x": [1, 2], "y": [3, 4], "z": [5, 6]}
    ).to_csv(csv, index=False)
    with pytest.warns(DeprecationWarning, match="atlas_labels"):
        cpm = CPMAnalysis(results_directory=str(tmp_path / "res"), atlas_labels=str(csv))
    assert len(cpm.atlas) == 2


def test_atlas_wins_over_atlas_labels(tmp_path):
    csv = tmp_path / "custom.csv"
    pd.DataFrame(
        {"region": ["a", "b"], "x": [1, 2], "y": [3, 4], "z": [5, 6]}
    ).to_csv(csv, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cpm = CPMAnalysis(
            results_directory=str(tmp_path / "res"),
            atlas="Power264",
            atlas_labels=str(csv),
        )
    assert len(cpm.atlas) == 264  # atlas took precedence


def test_node_count_mismatch_raises_on_run(tmp_path):
    # 100-region atlas but a connectome with 10 nodes (45 edges).
    cpm = CPMAnalysis(results_directory=str(tmp_path / "res"), atlas="Schaefer100-17")
    n_edges = 10 * 9 // 2
    X = np.random.rand(30, n_edges)
    y = np.random.rand(30)
    cov = np.random.rand(30, 1)
    with pytest.raises(ValueError, match="Atlas has 100 regions"):
        cpm.run(X=X, y=y, covariates=cov)
