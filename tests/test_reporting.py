"""
Smoke tests for the Jinja2 HTML report.

These run against the committed results fixture (no full pipeline needed) and
assert the report is generated, self-contained, and contains the expected
sections — with and without an atlas.
"""

import shutil
from pathlib import Path

import pytest

from cccpm.reporting.html_report import HTMLReporter

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_RESULTS = REPO_ROOT / "examples" / "results" / "regression_quickstart"
ATLAS_CSV = REPO_ROOT / "tests" / "fixtures" / "atlas_30.csv"


@pytest.fixture()
def results_dir(tmp_path):
    """A throwaway copy of the regression results fixture."""
    dst = tmp_path / "results"
    shutil.copytree(FIXTURE_RESULTS, dst)
    return dst


def _generate(results_dir, atlas=None):
    HTMLReporter(results_directory=str(results_dir), atlas_labels=atlas).generate_html_report()
    report = results_dir / "report.html"
    assert report.exists(), "report.html was not written"
    html = report.read_text(encoding="utf-8")
    assert len(html) > 10_000, "report is suspiciously small"
    return html


def test_report_generates_without_atlas(results_dir):
    html = _generate(results_dir)

    # Core sections present
    for needle in (
        'id="overview"',
        "Model Comparison",
        'id="predictions"',
        'id="brain-plots"',
        'id="stable-edges"',
        "Data &amp; Methods",
    ):
        assert needle in html, f"missing section/marker: {needle}"

    # Hero verdict + stat chips
    assert "The connectome model predicted" in html
    assert "stat-chip" in html

    # Self-contained vector figures, no leftover arakawa
    assert "<svg" in html
    assert "arakawa" not in html

    # Atlas-free graceful degradation: matrix + hubs render, chord does not
    assert "Connectivity matrix" in html
    assert "Hub nodes" in html
    assert "Chord diagram" not in html


def test_report_generates_with_atlas(results_dir):
    html = _generate(results_dir, atlas=str(ATLAS_CSV))

    # Atlas-dependent figures now appear
    assert "Connectivity matrix" in html
    assert "aggregated by canonical brain network" in html
    assert "Chord diagram" in html
