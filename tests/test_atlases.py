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

def test_bundled_atlases_have_required_schema():
    for name in list_atlases():
        df = load_atlas(name)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"{name} missing {col}"
        assert len(df) > 0
        for col in ("x", "y", "z"):
            assert pd.api.types.is_numeric_dtype(df[col]), f"{name}.{col} not numeric"


# What each bundled atlas must contain. Spans coordinate-defined, volumetric and
# surface-derived (volumetric-MNI) parcellations; one table rather than five
# near-identical test functions.
ATLAS_EXPECTATIONS = [
    # (name, n_regions, hemispheres, structures)
    ("Schaefer100-7", 100, None, None),
    ("Schaefer100-17", 100, None, None),
    ("Schaefer400-17", 400, None, None),
    ("Power264", 264, None, None),
    ("Dosenbach160", 160, None, None),
    ("Seitzman300", 300, None, None),
    ("Destrieux148", 148, {"L", "R"}, None),
    ("HarvardOxfordCortical", 96, {"L", "R"}, None),
    ("AAL116", 116, None, {"cortical", "subcortical", "cerebellum"}),
    ("Glasser360", 360, {"L", "R"}, None),
    ("DesikanKilliany68", 68, {"L", "R"}, None),
]


def test_bundled_atlas_contents():
    """One test over the table, not one per atlas: the failure message names the
    atlas, so per-atlas test IDs only inflate the collected count."""
    for name, n_regions, hemispheres, structures in ATLAS_EXPECTATIONS:
        df = load_atlas(name)
        assert len(df) == n_regions, f"{name} has {len(df)} regions"
        if hemispheres is not None:
            assert set(df["hemisphere"]) >= hemispheres, name
        if structures is not None:
            assert set(df["structure"]) >= structures, name


def test_expected_atlases_are_bundled():
    assert {name for name, *_ in ATLAS_EXPECTATIONS} <= set(list_atlases())


def test_schaefer_17_networks_are_finer_than_7():
    assert (load_atlas("Schaefer100-17")["network"].nunique()
            > load_atlas("Schaefer100-7")["network"].nunique())


def test_region_names_match_anatomy():
    """Names were assigned by anatomy; spot-check a landmark lands where it should."""
    dk = load_atlas("DesikanKilliany68")
    x, _, z = dk.loc[dk.region == "L_precentral", ["x", "y", "z"]].to_numpy()[0]
    assert x < 0 and z > 30  # left, dorsal -- motor strip


def test_wholebrain_variants_add_subcortical():
    """Each whole-brain variant = its cortical atlas + 14 subcortical structures."""
    for cortical, wholebrain in [
        ("DesikanKilliany68", "DesikanKillianyWholeBrain"),
        ("Destrieux148", "DestrieuxWholeBrain"),
        ("Glasser360", "GlasserWholeBrain"),
        ("HarvardOxfordCortical", "HarvardOxfordWholeBrain"),
    ]:
        wb = load_atlas(wholebrain)
        assert len(wb) == len(load_atlas(cortical)) + 14, wholebrain

        sub = wb[wb["structure"] == "subcortical"]
        assert len(sub) == 14, wholebrain
        assert {"L_Hippocampus", "R_Thalamus", "L_Amygdala"} <= set(sub["region"])
        # Subcortical structures are lateralised correctly (left is x < 0).
        assert ((sub["hemisphere"] == "L") == (sub["x"] < 0)).all(), wholebrain


def test_load_atlas_is_case_insensitive():
    assert len(load_atlas("schaefer100-17")) == len(load_atlas("Schaefer100-17"))


def test_load_unknown_atlas_raises():
    with pytest.raises(AtlasError, match="Unknown built-in atlas"):
        load_atlas("NotARealAtlas")


# ---------------------------------------------------------------------------
# resolve_atlas: path vs name vs None
# ---------------------------------------------------------------------------

def _custom_csv(tmp_path, name="custom.csv", **columns):
    csv = tmp_path / name
    pd.DataFrame(columns or {"region": ["a", "b"], "x": [1, 2],
                             "y": [3, 4], "z": [5, 6]}).to_csv(csv, index=False)
    return csv


def test_resolve_atlas_accepts_none_a_name_and_a_path(tmp_path):
    """The three things `atlas=` may be, in one place."""
    assert resolve_atlas(None) is None
    assert len(resolve_atlas("Power264")) == 264
    assert list(resolve_atlas(str(_custom_csv(tmp_path)))["region"]) == ["a", "b"]


def test_resolve_atlas_rejects_bad_paths(tmp_path):
    with pytest.raises(AtlasError, match="does not exist"):
        resolve_atlas(str(tmp_path / "nope.csv"))

    incomplete = _custom_csv(tmp_path, "bad.csv", region=["a"], x=[1])
    with pytest.raises(AtlasError, match="missing required column"):
        resolve_atlas(str(incomplete))


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
