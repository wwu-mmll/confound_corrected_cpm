import os
import itertools

from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns
from sciplotlib import style

from sklearn.model_selection import RepeatedKFold

from cpm.simulation.simulate_data_new_version import (simulate_scalar_X_with_target_corr,
                                                      build_multivariate_X_from_scalar, plot_scalar_X)
from cpm import CPMRegression
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection


# ----------------------------------------
# Set plot style
# ----------------------------------------
mpl.style.use(style.get_style("nature-reviews"))

def mm_to_inch(val_in_mm):
    return val_in_mm / 25.4

mpl.rc("xtick", labelsize=6)
mpl.rc("ytick", labelsize=6)
mpl.rc("axes", labelsize=6, titlesize=6)
mpl.rc("figure", dpi=450)

# Line and marker sizes
mpl.rc("lines", linewidth=0.5, markersize=0.25)

# Tick line width and length
mpl.rc("xtick.major", width=0.5, size=2)
mpl.rc("ytick.major", width=0.5, size=2)

# Spine (axis border) width
mpl.rc("axes", linewidth=0.5)

# ----------------------------------------
# Simulate different confounding scenarios
# ----------------------------------------
n_repeats = 10
n_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n_features = 105
n_signal_features = 15
n_confounds = 5
rho_target = 0.6
run_cpm = False
create_plots = True
edge_statistics = ['pearson', 'pearson_partial']
residualize = [False, True]
results_folder = "/home/nwinter/PycharmProjects/cpm_analyses/simulation_results"
data_scenarios = {
    'No Confounding Effect': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
    'Moderate Confounding Effect': {'alpha': 0.7, 'beta': 0.7, 'gamma': 0.7},
    'Strong Confounding Effect': {'alpha': 0.0, 'beta': 1.0, 'gamma': 1.0},
}

configurations = itertools.product(
    n_samples,
    range(n_repeats),
    residualize,
    edge_statistics,
    data_scenarios.keys()
)

# ----------------
# Run CPM analysis
# ----------------
for n, run, calculate_residuals, edge_statistic, scenario in configurations:
    if not run_cpm:
        break

    np.random.seed(run)
    folder = os.path.join(results_folder, f"n_samples={n}", f"{edge_statistic}" ,f"residuals_{calculate_residuals}", scenario, f'{run}')

    # simulate data
    X_scalar, y, z, signal = simulate_scalar_X_with_target_corr(
        n=n, alpha=data_scenarios[scenario]['alpha'], beta=data_scenarios[scenario]['beta'], gamma=data_scenarios[scenario]['gamma'],
        rho_target=rho_target
    )
    X_multi = build_multivariate_X_from_scalar(X_scalar, n_features=n_features, n_signal_features=n_signal_features)
    z_multi = build_multivariate_X_from_scalar(z, n_features=n_confounds, n_signal_features=n_confounds)

    #plot_scalar_X(X_scalar, y, z, scenario)

    # run CPM analysis
    univariate_edge_selection = UnivariateEdgeSelection(edge_statistic=edge_statistic,
                                                        edge_selection=[PThreshold(threshold=[0.05], correction=None)])
    cpm = CPMRegression(results_directory=folder,
                        cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=42),
                        edge_selection=univariate_edge_selection,
                        calculate_residuals=calculate_residuals,
                        n_permutations=2)
    cpm.run(X=X_multi, y=y, covariates=z_multi)


# -----------
# Create plot
# -----------
def load_results(configurations):
    # load results
    dfs = []
    for n, run, calculate_residuals, edge_statistic, scenario in configurations:
        df = pd.read_csv(os.path.join(results_folder, f"n_samples={n}",
                    edge_statistic, f"residuals_{calculate_residuals}", scenario, str(run), "cv_results.csv"))
        residuals_name = "Residualized " if calculate_residuals else ""
        df["analysis_type"] = f"{residuals_name}{edge_statistic.replace('_', ' ').replace('p', 'P')}"
        df["scenario"] = scenario
        df["n"] = n
        df = df[df["network"] == "both"].copy()
        dfs.append(df)
    df = pd.concat(dfs)
    df["analysis_type"] = pd.Categorical(df["analysis_type"], categories=['Pearson', 'Pearson Partial',
                                                                          'Residualized Pearson', 'Residualized Pearson Partial'], ordered=True)
    df["model"] = df["model"].replace({"covariates": "confounds", "full": "confounds+\nconnectome"})
    #df = df[df["model"] != "increment"]
    return df


if create_plots:
    metric = "explained_variance_score"

    model_plot_order = ["connectome", "confounds+\nconnectome", "residuals", "increment"]
    color_map = {"Pearson": "#113F67", "Pearson Partial": "#34699A",
                 "Residualized Pearson": "#D5451B", "Residualized Pearson Partial": "#FF9B45",
                 "confounds": "#393E46"}
    analysis_order = ['Pearson', 'Pearson Partial', 'Residualized Pearson', 'Residualized Pearson Partial']

    # load results
    df = load_results(configurations)

    for n in n_samples:
        df_n = df[df["n"] == n]

        # create plot
        n_rows = len(data_scenarios)
        fig = plt.figure(figsize=(mm_to_inch(180), mm_to_inch(30 * n_rows)))
        gs = gridspec.GridSpec(n_rows, 8, width_ratios=[1, 3, 0.8, 1, 0.4, 1, 0.4, 1], figure=fig)  # 1:3 width ratio
        gs.update(hspace=1, wspace=0)

        axes_conf = [fig.add_subplot(gs[row, 0]) for row in range(n_rows)]
        axes_models = [fig.add_subplot(gs[row, 1]) for row in range(n_rows)]

        handles, labels = None, None

        for row_idx, scenario in enumerate(data_scenarios.keys()):
            cdf = df_n[df_n["scenario"] == scenario]
            # === Column 1: Confounds only ===
            ax0 = axes_conf[row_idx]
            df_conf = cdf[cdf["model"] == "confounds"]

            sns.barplot(
                data=df_conf,
                x="model",
                y=metric,
                ax=ax0,
                errorbar='sd',
                width=0.2,
                color=color_map['confounds'],
                linewidth=0.5
            )
            ax0.set_ylabel(r"$R^2$")
            ax0.set_xlabel("")
            ax0.set_title(scenario, fontweight="bold", loc="left")
            ax0.set_ylim(0, 0.5)
            ax0.set_yticks([0, 0.25, 0.5])

            # === Columns 2-4: All 3 models × grouped analysis types ===
            ax2 = axes_models[row_idx]
            df_subset = cdf[cdf["model"].isin(model_plot_order)]

            sns.barplot(
                data=df_subset,
                x="model",
                y=metric,
                hue="analysis_type",
                hue_order=analysis_order,
                order=model_plot_order,
                palette=color_map,
                ax=ax2,
                errorbar='sd',
                linewidth=0.5
            )

            if row_idx == 0:
                handles, labels = ax2.get_legend_handles_labels()
            ax2.get_legend().remove()

            ax2.set_ylim(0, 0.5)
            ax2.get_yaxis().set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.set_xlabel("")

            X_scalar, y, z, signal = simulate_scalar_X_with_target_corr(
                n=n, alpha=data_scenarios[scenario]['alpha'], beta=data_scenarios[scenario]['beta'],
                gamma=data_scenarios[scenario]['gamma'],
                rho_target=rho_target
            )
            X_multi = build_multivariate_X_from_scalar(X_scalar, n_features=n_features, n_signal_features=n_signal_features)
            z_multi = build_multivariate_X_from_scalar(z, n_features=n_confounds, n_signal_features=n_confounds)
            from sklearn.linear_model import LinearRegression
            from scipy.stats import pearsonr


            def partial_correlation(x, y, z):
                x_resid = x - LinearRegression().fit(z.reshape(-1, 1), x).predict(z.reshape(-1, 1))
                return pearsonr(x_resid, y)[0], x_resid


            r_xy = pearsonr(X_scalar, y)[0]
            r_xz = pearsonr(X_scalar, z)[0]
            r_yz = pearsonr(y, z)[0]
            r_partial, x_resid = partial_correlation(X_scalar, y, z)

            df_sim = pd.DataFrame({'connectome': X_scalar, 'y': y, 'confounds': z, 'res(connectome)': x_resid})

            sax1 = fig.add_subplot(gs[row_idx, 5])
            sns.regplot(y='y', x='connectome', data=df_sim, ax=sax1, marker='o', line_kws={'alpha': 1, 'color': 'k'}, scatter_kws={'alpha': 1,
                                                                          's':2,'edgecolors': '#004030', 'linewidths': 0.4,
                                                                                                                                  'color': '#3F7D58'})
            sax1.set_title(f'r={r_xy:.2f}', fontsize=6)
            sax1.tick_params(axis='y', which='both', left=False, labelleft=False)

            sax1.set_xlim(-3, 3)
            sax1.set_xticks([-3, 0, 3])
            sax1.set_ylim(-3, 3)
            if row_idx < 2:
                sax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            sax2 = fig.add_subplot(gs[row_idx, 3])
            sns.regplot(x='confounds', y='y', data=df_sim, ax=sax2, marker='o', line_kws={'alpha': 1, 'color': 'k'}, scatter_kws={'alpha': 1,
                                                                          's':2,'edgecolors': '#004030', 'linewidths': 0.4,
                                                                                                                                  'color': '#3F7D58'})
            sax2.set_title(f'r={r_yz:.2f}', fontsize=6)

            sax2.set_xlim(-3, 3)
            sax2.set_xticks([-3, 0, 3])
            sax2.set_ylim(-3, 3)
            sax2.set_yticks([-3, 0, 3])
            if row_idx < 2:
                sax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            sax3 = fig.add_subplot(gs[row_idx, 7])
            sns.regplot(y='y', x='res(connectome)', data=df_sim, ax=sax3,  marker='o', line_kws={'alpha': 1, 'color': 'k'}, scatter_kws={'alpha': 1,
                                                                          's':2,'edgecolors': '#004030', 'linewidths': 0.4,
                                                                                                                                  'color': '#3F7D58'})
            sax3.set_title(f'r={r_partial:.2f}', fontsize=6)
            sax3.tick_params(axis='y', which='both', left=False, labelleft=False)
            sax3.set_xlim(-3, 3)
            sax3.set_xticks([-3, 0, 3])
            sax3.set_ylim(-3, 3)
            if row_idx < 2:
                sax3.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # --- Global Legend ---
        fig.legend(
            handles,
            labels,
            title="",
            loc="lower center",
            bbox_to_anchor=(0.32, -0.0),  # center below figure
            ncol=2,                       # split into 2 columns (2 rows if 4 items)
            frameon=False,
            fontsize=6
        )

        #plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        plt.savefig(os.path.join(results_folder, f"n_samples={n}_simulation_results.pdf"))
        plt.show()