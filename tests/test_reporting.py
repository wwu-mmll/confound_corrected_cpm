"""Smoke tests for the Jinja2 HTML report, against a committed results
fixture (no full pipeline needed), with and without an atlas."""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import report_text

from cccpm.reporting.html_report import HTMLReporter
from cccpm.reporting.plots.connectome_utils import masked_signed_stability_matrix

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_RESULTS = REPO_ROOT / "examples" / "results" / "regression_quickstart"
# Must match the atlas the fixture was generated with, or region counts mismatch.
ATLAS_CSV = REPO_ROOT / "src" / "cccpm" / "atlases" / "data" / "Schaefer100-17.csv"


@pytest.fixture()
def results_dir(tmp_path):
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

    for needle in (
        'id="overview"',
        "Model Comparison",
        'id="network-strengths"',
        'id="brain-plots"',
        'id="stable-edges"',
        "Data &amp; Methods",
    ):
        assert needle in html, f"missing section/marker: {needle}"

    assert "The connectome model predicted" in html
    assert "stat-chip" in html
    assert "scatter-row" in html
    assert "Connectome — positive network" in html
    assert "Connectome — negative network" in html
    assert "Covariates only" in html

    assert "<svg" in html
    assert "arakawa" not in html  # no leaked plotting-lib signature

    # Figure captions (not the explanatory text, which names every figure):
    # atlas-free degrades gracefully -- matrix + hubs render, chord does not.
    assert "Connectivity matrix (" in html
    assert "Hub nodes —" in html
    assert "Chord diagram of between-network connectivity" not in html

    assert 'class="brain-controls"' in html
    assert 'data-mode="top5"' in html
    assert 'data-mode="top10"' in html


def test_report_generates_with_atlas(results_dir):
    html = _generate(results_dir, atlas=str(ATLAS_CSV))

    assert "Connectivity matrix (" in html
    assert "aggregated by canonical brain network" in html
    assert "Chord diagram of between-network connectivity" in html


def test_masked_signed_stability_matrix_modes():
    n = 4
    stab = np.zeros((n, n, 2))
    stab[0, 1, 0] = 1.0   # positive network
    stab[0, 2, 0] = 0.6
    stab[0, 3, 0] = 0.2
    stab[1, 2, 1] = 0.9   # negative network

    sig = np.ones((n, n, 2))
    sig[0, 1, 0] = 0.01   # significant
    sig[0, 2, 0] = 0.20
    sig[0, 3, 0] = 0.30
    sig[1, 2, 1] = 0.02   # significant

    # all: every stable edge, signed by network (positive +, negative -)
    all_m = masked_signed_stability_matrix(stab, sig, mode="all")
    assert all_m[0, 1] == 1.0
    assert all_m[0, 2] == 0.6
    assert all_m[1, 2] == -0.9

    sig_m = masked_signed_stability_matrix(stab, sig, mode="significant")
    assert sig_m[0, 1] == 1.0
    assert sig_m[0, 2] == 0.0
    assert sig_m[1, 2] == -0.9

    # top10: within the positive layer only the most stable edge survives
    top_m = masked_signed_stability_matrix(stab, sig, mode="top10")
    assert top_m[0, 1] == 1.0
    assert top_m[0, 2] == 0.0
    assert top_m[1, 2] == -0.9  # sole negative edge is its own top

    # No significance data: falls back to all stable edges.
    nosig = masked_signed_stability_matrix(stab, None, mode="significant")
    assert nosig[0, 2] == 0.6


def test_report_text_leaves_no_base64_for_a_word_search_to_trip_over(results_dir):
    """`report_text` must return prose, not image bytes: other tests grep this
    text for words, and leftover base64 makes a hit/miss a coin flip decided
    by rendered pixels. Regression: the old stripper missed matplotlib's SVG
    output (newline after the comma), so macOS/Windows CI went red while Linux
    stayed green."""
    _generate(results_dir)
    text = report_text(results_dir)

    assert "base64" not in text
    # Long unbroken alphanumeric runs are the signature of an embedded payload;
    # no word in the report's prose comes close.
    longest = max((len(run) for run in re.findall(r"[A-Za-z0-9+/=]+", text)), default=0)
    assert longest < 60, f"a {longest}-character run survived -- something is still embedded"
