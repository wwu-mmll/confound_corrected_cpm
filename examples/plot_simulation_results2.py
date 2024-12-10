import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


results_folder = "simulated_data"
edge_statistics = ['pearson', 'pearson_partial']


for scenario in ['A', 'B']:
    for edge_statistic in edge_statistics:
        results = {f"scenario_{scenario}1": os.path.join(results_folder, f"scenario_{scenario}1", "results", edge_statistic),
                   f"scenario_{scenario}2": os.path.join(results_folder, f"scenario_{scenario}2", "results", edge_statistic),
                   f"scenario_{scenario}3": os.path.join(results_folder, f"scenario_{scenario}3", "results", edge_statistic),
                   f"scenario_{scenario}4": os.path.join(results_folder, f"scenario_{scenario}4", "results", edge_statistic),
                   }

        dfs = []
        for link_type, folder in results.items():
            df = pd.read_csv(os.path.join(folder, 'cv_results.csv'))
            df['link_type'] = link_type
            dfs.append(df)

        concatenated_df = pd.concat(dfs, ignore_index=True)

        concatenated_df = concatenated_df[concatenated_df['network'] == 'both']


        g = sns.FacetGrid(concatenated_df, col="link_type", margin_titles=True, despine=True,
                          height=2.5, hue="model")
        g.map(plt.axvline, x=0, color='grey', linewidth=0.5, zorder=-1)
        g.map(sns.violinplot, "pearson_score", "model", inner=None, split=True,# hue="model",# hue_order=[1, 2],
              density_norm='count', dodge=True, palette="Blues_r", fill=True)
        #g.map(sns.boxplot, "pearson_score", "model", dodge=True, #hue=1, hue_order=[2, 1]
        #      )
        g.set_titles(col_template="{col_name}", size=7)
        g.set_xlabels("Pearson correlation", size=8)
        plt.suptitle(edge_statistic)
        plt.show()
        print()
