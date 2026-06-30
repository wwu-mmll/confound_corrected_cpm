"""
Fast iteration loop for HTML report development.

Regenerates report.html from a fixed results fixture and opens it in the
default browser so visual edits to templates/styles.css or section_builders.py
can be previewed in a few seconds.

Usage:
    poetry run python scripts/preview_report.py
    poetry run python scripts/preview_report.py --results examples/results/classification_quickstart
    poetry run python scripts/preview_report.py --open  # open after generation (default)
    poetry run python scripts/preview_report.py --no-open
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

# Ensure the src layout is importable when running from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cccpm.reporting.html_report import HTMLReporter


DEFAULT_RESULTS = REPO_ROOT / "examples" / "results" / "regression_quickstart"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the CCCPM HTML report.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to a CPM results directory (default: examples/results/regression_quickstart)",
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help="Optional path to atlas labels CSV",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        default=True,
        help="Open report in browser after generation (default: on)",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
    )
    args = parser.parse_args()

    results_dir = args.results.resolve()
    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    atlas = str(args.atlas) if args.atlas else None

    print(f"Generating report from: {results_dir}")
    reporter = HTMLReporter(results_directory=str(results_dir), atlas_labels=atlas)
    reporter.generate_html_report()

    report_path = results_dir / "report.html"
    print(f"Report written to: {report_path}")

    if args.open_browser:
        webbrowser.open(report_path.as_uri())


if __name__ == "__main__":
    main()
