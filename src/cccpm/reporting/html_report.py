"""
HTML Report Generator for Confound-Corrected CPM Analysis.

Builds a self-contained single-page HTML report using a Jinja2 template.
The public interface (HTMLReporter / generate_html_report) is unchanged.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jinja2

import cccpm
from cccpm.reporting.data_loader import ReportDataLoader
from cccpm.reporting.reporting_utils import embed_image_base64
from cccpm.reporting.section_builders import (
    build_brain_plots_context,
    build_data_context,
    build_network_strengths_context,
    build_overview_context,
    build_performance_context,
    build_stable_edges_context,
)


class HTMLReporter:
    """
    Generate a self-contained HTML report from CPM analysis results.

    The report is a single scrolling page with a sticky sidebar; it embeds
    all figures as base64 data URIs so it is fully portable and works offline.

    Args:
        results_directory: Path to the directory containing CPM results.
        atlas_labels: Optional path to a CSV file with atlas region labels
            (enables brain plots and labelled edge tables).
    """

    def __init__(self, results_directory: str, atlas_labels: str | None = None):
        self.results_directory = results_directory
        self.plots_dir = os.path.join(results_directory, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        self.data_loader = ReportDataLoader(results_directory, atlas_labels)
        self.X_names, self.y_name, self.covariates_names = (
            self.data_loader.load_variable_names()
        )
        self.atlas_labels = self.data_loader.atlas_labels
        self.task_type = self.data_loader.load_task_type()
        self._load_data()

    def _load_data(self) -> None:
        self.df, self.df_mean = self.data_loader.load_cv_results()
        self.df_predictions = self.data_loader.load_predictions()
        self.df_p_values = self.data_loader.load_p_values()
        self.df_permutations = self.data_loader.load_permutations()
        self.df_network_strengths = self.data_loader.load_network_strengths()

    def generate_html_report(self) -> None:
        """Render the HTML report and write it to ``results_directory/report.html``."""
        summary_df, scatter_path, _ = self.data_loader.load_data_insights()
        edge_stability, edge_stability_sig = self.data_loader.load_edge_stability()

        # ── Build template context ──────────────────────────────────────────
        ctx: dict = {
            "y_name": self.y_name,
            "logo_img": self._logo_html(),
        }

        ctx.update(build_overview_context(
            results_directory=self.results_directory,
            version=cccpm.__version__,
            run_date=datetime.date.today().isoformat(),
        ))

        ctx.update(build_data_context(summary_df, scatter_path))

        ctx.update(build_performance_context(
            df_full=self.df,
            df_mean=self.df_mean,
            df_p_values=self.df_p_values,
            df_predictions=self.df_predictions,
            y_name=self.y_name,
            task_type=self.task_type,
            plots_dir=self.plots_dir,
        ))

        ctx.update(build_network_strengths_context(
            df_network_strengths=self.df_network_strengths,
            y_name=self.y_name,
            plots_dir=self.plots_dir,
        ))

        ctx.update(build_brain_plots_context(
            results_directory=self.results_directory,
            plots_dir=self.plots_dir,
            atlas_labels=self.atlas_labels,
            edge_stability=edge_stability,
        ))

        ctx.update(build_stable_edges_context(
            edge_stability=edge_stability,
            edge_stability_significance=edge_stability_sig,
            atlas_labels=self.atlas_labels,
        ))

        # ── Render ─────────────────────────────────────────────────────────
        html = self._render(ctx)

        out_path = os.path.join(self.results_directory, "report.html")
        Path(out_path).write_text(html, encoding="utf-8")

        plt.close("all")

    # ── Private helpers ────────────────────────────────────────────────────

    def _logo_html(self) -> str:
        logo_path = Path(__file__).parent / "assets" / "CCCPM.png"
        if logo_path.exists():
            return embed_image_base64(str(logo_path))
        return ""

    def _render(self, ctx: dict) -> str:
        template_dir = Path(__file__).parent / "templates"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        # Disable autoescape for HTML fragment variables (figures, tables)
        env.autoescape = False

        template = env.get_template("report.html.j2")

        # Inline the stylesheet
        css_path = template_dir / "styles.css"
        ctx["styles"] = css_path.read_text(encoding="utf-8")

        return template.render(**ctx)
