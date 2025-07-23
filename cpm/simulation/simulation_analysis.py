import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold

from cpm.simulation.simulate_data_new_version import (simulate_scalar_X_with_target_corr,
                                                      build_multivariate_X_from_scalar, plot_scalar_X,
                                                      simulate_independent_sources_predicting_y)
from cpm import CPMRegression
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


# -----------------------
# Simulate all 3 scenarios
# -----------------------
n_repeats = 10
n = 1000
n_features = 105
n_signal_features = 15
rho_target = 0.6
run_cpm = False
investigate = True
edge_statistics = ['pearson', 'pearson_partial']
residualize = [False, True]
results_folder = "simulation_results"
data_scenarios = {
    'Direct Only': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'name': "No Confounding Effect"},
    'Independent Contribution': {'name': "Independent Contribution"},
    'Mixed Direct + Confounding': {'alpha': 0.7, 'beta': 0.7, 'gamma': 0.7, 'name': "Some Confounding Effect"},
    'Confounding Only': {'alpha': 0.0, 'beta': 1.0, 'gamma': 1.0, 'name': "Only Confounding Effect"},
}

if run_cpm:
    for run in range(n_repeats):
        for edge_statistic in edge_statistics:
            for calculate_residuals in residualize:
                for scenario, params in data_scenarios.items():
                    np.random.seed(run)
                    folder = os.path.join(results_folder, f"n_samples={n}", f"{edge_statistic}" ,f"residuals_{calculate_residuals}", scenario, f'{run}')

                    if scenario == 'Independent Contribution':
                        X_scalar, y, z = simulate_independent_sources_predicting_y(
                            n=n, theta_x=1.0, theta_z=1.0, noise_y=1.0, noise_x=1.0, noise_z=1.0
                        )
                        X_multi = build_multivariate_X_from_scalar(X_scalar, n_features=n_features, n_signal_features=n_signal_features)
                        z_multi = build_multivariate_X_from_scalar(z, n_features=5, n_signal_features=5)
                        plot_scalar_X(X_scalar, y, z, scenario)
                    else:
                        # simulate data
                        X_scalar, y, z, signal = simulate_scalar_X_with_target_corr(
                            n=n, alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'],
                            rho_target=rho_target
                        )
                        X_multi = build_multivariate_X_from_scalar(X_scalar, n_features=n_features, n_signal_features=n_signal_features)
                        z_multi = build_multivariate_X_from_scalar(z, n_features=5, n_signal_features=5)

                        plot_scalar_X(X_scalar, y, z, scenario)

                    # run CPM analysis
                    univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=edge_statistic,
                                                                        edge_selection=[PThreshold(threshold=[0.05],
                                                                                                   correction=None)])

                    cpm = CPMRegression(results_directory=folder,
                                        cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=42),
                                        edge_selection=univariate_edge_selection,
                                        # cv_edge_selection=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
                                        calculate_residuals=calculate_residuals,
                                        n_permutations=2)
                    cpm.run(X=X_multi, y=y, covariates=z_multi)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sciplotlib import style

mpl.style.use(style.get_style("nature-reviews"))


def mm_to_inch(val_in_mm):
    return val_in_mm / 25.4

mpl.rc("xtick", labelsize=11)
mpl.rc("ytick", labelsize=11)
mpl.rc("axes", labelsize=12, titlesize=12)
mpl.rc("figure", dpi=450)
mpl.rc("lines", linewidth=1, markersize=2)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
metric = "explained_variance_score"

model_rename = {
    "covariates": "confounds",
    "full": "confounds+\nconnectome"
}

analysis_rename = {
    "pearson+resid_False": "Pearson",
    "pearson_partial+resid_False": "Partial Pearson",
    "pearson+resid_True": "Residualized Pearson",
    "pearson_partial+resid_True": "Residualized Partial Pearson"
}

analysis_order = [
    "Pearson",
    "Partial Pearson",
    "Residualized Pearson",
    "Residualized Partial Pearson"
]

model_plot_order = ["connectome", "confounds+\nconnectome", "residuals"]

color_map = {
    "Pearson": "#113F67",
    "Partial Pearson": "#34699A",
    "Residualized Pearson": "#D5451B",
    "Residualized Partial Pearson": "#FF9B45"
}

def mm_to_inch(mm):
    return mm / 25.4

def load_all_runs(scenario):
    dfs = []
    for run in range(n_repeats):
        for residual in residualize:
            for stat in edge_statistics:
                folder = os.path.join(
                    results_folder,
                    f"n_samples={n}",
                    stat,
                    f"residuals_{residual}",
                    scenario,
                    str(run)
                )
                df = pd.read_csv(os.path.join(folder, "cv_results.csv"))
                df["analysis_type"] = f"{stat}+resid_{residual}"
                dfs.append(df)
    return pd.concat(dfs)

# --- Main Plot ---
if investigate:
    n_rows = len(data_scenarios)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=4,
        figsize=(mm_to_inch(240), mm_to_inch(40 * n_rows)),
        sharey=True
    )

    if n_rows == 1:
        axes = [axes]  # ensure 2D-like access even with one row

    handles, labels = None, None

    for row_idx, (scenario, params) in enumerate(data_scenarios.items()):
        df = load_all_runs(scenario)
        df = df[df["network"] == "both"].copy()
        df = df[df["model"] != "increment"]

        # Rename for plotting
        df["analysis_type"] = df["analysis_type"].replace(analysis_rename)
        df["analysis_type"] = pd.Categorical(df["analysis_type"], categories=analysis_order, ordered=True)
        df["model"] = df["model"].replace(model_rename)

        # === Column 1: Confounds only ===
        ax0 = axes[row_idx][0]
        df_conf = df[df["model"] == "confounds"]

        sns.barplot(
            data=df_conf,
            x="model",
            y=metric,
            hue="analysis_type",
            hue_order=analysis_order,
            palette=color_map,
            ax=ax0,
            dodge=True
        )
        ax0.set_title("Confounds", fontweight="bold")
        ax0.set_ylabel(r"$R^2$" if row_idx == 0 else "")
        ax0.set_xlabel("")
        if row_idx == 0:
            handles, labels = ax0.get_legend_handles_labels()
        ax0.get_legend().remove()

        # === Columns 2-4: All 3 models × grouped analysis types ===
        for col_idx, analysis_label in enumerate(analysis_order, start=1):
            ax = axes[row_idx][col_idx]
            df_subset = df[df["model"].isin(model_plot_order)]

            sns.barplot(
                data=df_subset,
                x="model",
                y=metric,
                hue="analysis_type",
                hue_order=analysis_order,
                order=model_plot_order,
                palette=color_map,
                ax=ax,
                dodge=True
            )

            if row_idx == 0 and col_idx == 1:
                handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            if row_idx == 0:
                ax.set_title(analysis_label, fontweight="bold")
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20)

    # --- Global Legend ---
    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="",
            loc="upper right",
            bbox_to_anchor=(1.0, 0.95),
            borderaxespad=0
        )

    plt.tight_layout()
    plt.show()