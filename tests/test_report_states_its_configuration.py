"""
A report must say which analysis produced it.

Confound control is two run-level choices (`selection_input`, `model_input`), so
one report is one cell of that 2x2 and there is nothing to compare side by side
inside it. That makes it the report's job to say which cell it is. Before this,
a naive run and a fully confound-controlled run opened with the *identical*
sentence -- "The connectome model predicted target: r = ..." -- and the
configuration was only in a table at the bottom of the appendix. Anyone handed
the HTML could not tell them apart.
"""
import contextlib
import io
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import KFold

from conftest import report_text

from cccpm import CPMAnalysis, PThreshold, UnivariateEdgeSelection
from cccpm.reporting.reporting_utils import parse_config_block
from cccpm.reporting.section_builders import describe_confound_control


CELLS = list(itertools.product(['raw', 'residualized'], repeat=2))


def _run(results_dir, selection_input, model_input, covariates=True, n_permutations=0):
    rng = np.random.RandomState(0)
    n, n_nodes = 150, 10
    n_features = n_nodes * (n_nodes - 1) // 2
    X = rng.randn(n, n_features)
    Z = rng.randn(n, 3)
    y = X[:, :8].sum(1) * 0.3 + Z[:, 0] * 1.2 + rng.randn(n)

    ue = UnivariateEdgeSelection(
        selection_statistic="pearson", selection_input=selection_input,
        edge_selection=[PThreshold(threshold=0.05, correction=[None])])
    cpm = CPMAnalysis(
        results_directory=str(results_dir),
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        edge_selection=ue, model_input=model_input,
        n_permutations=n_permutations, task_type="regression")
    with contextlib.redirect_stdout(io.StringIO()):
        cpm.run(X=X, y=y, covariates=Z if covariates else None)
    return str(results_dir)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Run each configuration at most once, and share its results directory.

    The tests in this file need five distinct configurations and between them
    used to ask for seventeen analyses -- the four 2x2 cells alone were run
    three times over. Every test here only ever *reads* from the directory
    (run_config.json, report.html, cpm_log.txt), so one run per configuration
    is enough.
    """
    cache = {}

    def get(selection_input, model_input, covariates=True):
        key = (selection_input, model_input, covariates)
        if key not in cache:
            name = f"{selection_input}_{model_input}" + ("" if covariates else "_nocov")
            cache[key] = _run(tmp_path_factory.mktemp(name),
                              selection_input, model_input, covariates=covariates)
        return cache[key]

    return get


def _rendered_text(results_dir):
    """Report text with markup and embedded figures stripped.

    Thin alias for ``conftest.report_text`` -- base64 is not prose, and the
    report is UTF-8 whatever the runner's locale.
    """
    return report_text(results_dir)


def _headline(results_dir):
    """Just the headline callout.

    Deliberately not the flattened page text: the confound configuration is also
    explained in a panel further down, so searching the whole document would
    pass even with the headline saying nothing -- which is exactly what this
    test is for. (Checked: it does pass that way if you are not careful.)
    """
    html = Path(results_dir, 'report.html').read_text(encoding='utf-8')
    match = re.search(r'<div class="headline-callout">(.*?)</div>', html, re.S)
    assert match, "headline callout not found in the report"
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", match.group(1))).strip()


# ---------------------------------------------------------------------------
# The configuration is recorded, and recorded independently of the log
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selection_input,model_input", CELLS)
def test_run_config_is_persisted(run_dir, selection_input, model_input):
    """Written as a file, not parsed back out of cpm_log.txt: a run with logging
    turned down must still produce a report that knows its own configuration."""
    d = run_dir(selection_input, model_input)
    with open(os.path.join(d, 'run_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    assert cfg['selection_input'] == selection_input
    assert cfg['model_input'] == model_input
    assert cfg['has_covariates'] is True


def test_unknown_configuration_says_nothing_rather_than_guessing():
    """An old results directory has no run_config.json. Claiming "none" for a
    run that was in fact controlled would be worse than staying silent."""
    assert describe_confound_control({}) == (None, None)
    assert describe_confound_control(None) == (None, None)


# ---------------------------------------------------------------------------
# ...and it reaches the reader
# ---------------------------------------------------------------------------

def test_the_four_cells_produce_four_distinguishable_headlines(run_dir):
    headlines = {}
    for selection_input, model_input in CELLS:
        headline = _headline(run_dir(selection_input, model_input))
        assert "The connectome model predicted" in headline
        match = re.search(r"Confound control: ([a-z][a-z ]+)\.", headline)
        assert match, f"the headline does not name the configuration: {headline}"
        headlines[(selection_input, model_input)] = match.group(1).strip()

    assert len(set(headlines.values())) == 4, (
        f"cells are not distinguishable from the headline: {headlines}")
    assert headlines[('raw', 'raw')] == "none"
    assert headlines[('residualized', 'residualized')] == "edge selection and features"


@pytest.mark.parametrize("selection_input,model_input", CELLS)
def test_confound_control_is_explained_in_the_body(run_dir, selection_input, model_input):
    text = _rendered_text(run_dir(selection_input, model_input))
    label, explanation = describe_confound_control(
        {'selection_input': selection_input, 'model_input': model_input,
         'has_covariates': True})
    assert label and explanation
    # The explanation is prose; check a distinctive fragment survived rendering.
    assert explanation.split(" -- ")[0][:45] in text


def test_glossary_describes_the_connectome_model_this_run_produced(run_dir):
    """`connectome` means different things under the two model_input settings,
    and carries the same row label either way."""
    raw = _rendered_text(run_dir("residualized", "raw"))
    residualized = _rendered_text(run_dir("residualized", "residualized"))
    assert "deconfounded network strengths" in residualized
    assert "deconfounded network strengths" not in raw


def test_suppressed_increment_is_explained(run_dir):
    """The em dash in the Pearson column is deliberate; say so, or it reads as a
    failed computation."""
    text = _rendered_text(run_dir("residualized", "residualized"))
    assert "Fisher" in text and "Steiger" in text


def test_report_without_covariates_says_so(run_dir):
    d = run_dir("raw", "raw", covariates=False)
    with open(os.path.join(d, 'run_config.json'), encoding='utf-8') as f:
        assert json.load(f)['has_covariates'] is False
    assert "no covariates" in _rendered_text(d)


# ---------------------------------------------------------------------------
# The configuration table
# ---------------------------------------------------------------------------

def test_wrapped_config_values_do_not_become_their_own_rows(run_dir):
    """An estimator repr is logged over several indented lines. Treating the
    continuations as new keys produced rows like "threshold=[0.05])]," with an
    empty value."""
    d = run_dir("residualized", "residualized")
    pairs = parse_config_block(os.path.join(d, 'cpm_log.txt'))

    keys = [k for k, _ in pairs]
    assert "Edge selection method" in keys
    for key, value in pairs:
        assert value != "" or not key.endswith((",", ")", "]")), (
            f"continuation line became a row: {key!r}")

    edge_selection = dict(pairs)["Edge selection method"]
    assert "selection_input='residualized'" in edge_selection
    assert "selection_statistic='pearson'" in edge_selection
