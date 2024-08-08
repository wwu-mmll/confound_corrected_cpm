import streamlit as st
from streamlit_utils import style_apa
import pandas as pd
from plots.plots import plot_histogram_with_line
import os
import seaborn as sns


st.title('Permutation Results')


tab2, tab1 = st.tabs(["p-Values", "Permutations"])

with tab1:
    col1, col2 = st.columns([3, 1])
    # Style the DataFrame
    p_values = st.session_state.df_p_values
    p_values.set_index(['network', 'model'], inplace=True)
    styled_df = style_apa(p_values)

    with col2:
        with st.container(border=1):
            selected_metric = st.radio("Select a metric to display:", list(st.session_state.df.columns)[3:-1])
    with col1:
        long_perms = st.session_state.df_permutations.melt(id_vars=['network', 'model'], var_name='metric',
                                                        value_name='permuted_value')
        true_results = st.session_state.df_main
        true_results = true_results.loc[:, true_results.columns.get_level_values(1) == 'mean']
        true_results.columns = true_results.columns.droplevel(1)

        true_melted = true_results.reset_index().melt(id_vars=['network', 'model'], var_name='metric',
                                                      value_name='true_value')
        merged = pd.merge(long_perms, true_melted, on=['network', 'model', 'metric'])

        # Filter data based on selected metric
        metric_data = merged[merged['metric'] == selected_metric]

        g = sns.FacetGrid(metric_data, row="network", col="model", margin_titles=True, sharex=False,
                          sharey=False, height=2, aspect=1.5)

        # Map the plotting function to each subset
        g.map_dataframe(plot_histogram_with_line)

        # Adjust plot titles and labels
        g.set_titles(col_template="{col_name}", row_template="{row_name}", size=14)
        g.set_axis_labels(f"{selected_metric}", 'Count')

        # Display the plot
        plot_name = os.path.join(st.session_state.results_directory, "plots", 'permutations.png')
        g.fig.savefig(plot_name, dpi=300)

        st.image(plot_name, width=1000)
    # st.pyplot(g.fig)
with tab2:
    # Display in Streamlit
    st.write(styled_df.to_html(), unsafe_allow_html=True)