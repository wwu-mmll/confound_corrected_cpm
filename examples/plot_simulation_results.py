import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


results_folder = "/spm-data/vault-data3/mmll/projects/confound_corrected_cpm/results"

edge_statistics = ['pearson', 'pearson_partial']


for edge_statistic in edge_statistics:
    results = {"brain not associated with y\n brain not associated with confound\n confound associated with y": os.path.join(
                    results_folder, f"simulated_data_no_no_link_{edge_statistic}"),
        "brain not associated with y\n brain associated with confound\n confound associated with y": os.path.join(
            results_folder, f"simulated_data_no_link_{edge_statistic}"),
        "brain weakly associated with y\n brain associated with confound\n confound associated with y": os.path.join(
            results_folder, f"simulated_data_weak_link_{edge_statistic}"),
        "brain strongly associated with y\n brain associated with confound\n confound associated with y": os.path.join(
            results_folder, f"simulated_data_direct_link_{edge_statistic}")}
    dfs = []
    for link_type, folder in results.items():
        df = pd.read_csv(os.path.join(folder, 'cv_results.csv'))
        df['link_type'] = link_type
        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)

    concatenated_df = concatenated_df[concatenated_df['network'] == 'both']

    concatenated_df['model'] = concatenated_df['model'].replace({"covariates": "confound only", "full": "connectome + confound",
                                     "connectome": "connectome only"})


    g = sns.FacetGrid(concatenated_df, col="link_type", margin_titles=True, despine=True,
                      height=2.5, hue="model")
    g.map(plt.axvline, x=0, color='grey', linewidth=0.5, zorder=-1)
    g.map(sns.violinplot, "pearson_score", "model", inner=None, split=True,# hue="model",# hue_order=[1, 2],
          density_norm='count', dodge=True, palette="Blues_r", fill=True)
    #g.map(sns.boxplot, "pearson_score", "model", dodge=True, #hue=1, hue_order=[2, 1]
    #      )
    g.set_titles(col_template="{col_name}", size=7)
    g.set_xlabels("Pearson correlation", size=8)
    plt.show()
    print()
