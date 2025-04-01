import os

import pandas as pd
import streamlit as st

from streamlit_utils import load_results_from_folder, load_data_from_folder, style_apa


st.set_page_config(layout="wide")


def main():
    if 'results_directory' not in st.session_state:
        st.session_state[
            'results_directory'] = '/spm-data/vault-data3/mmll/projects/cpm_python/results_new/hcp_SSAGA_TB_Yrs_Smoked_spearman_partial_p=0.01'


    st.session_state['df'] = pd.read_csv(os.path.join(st.session_state.results_directory, 'cv_results.csv'))
    st.session_state['df_main'] = load_results_from_folder(st.session_state.results_directory, 'cv_results_mean_std.csv')
    st.session_state['df_predictions'] = load_data_from_folder(st.session_state.results_directory, 'cv_predictions.csv')
    st.session_state['df_p_values'] = load_data_from_folder(st.session_state.results_directory, 'p_values.csv')
    st.session_state['df_permutations'] = load_data_from_folder(st.session_state.results_directory, 'permutation_results.csv')

    os.makedirs(os.path.join(st.session_state.results_directory, "plots"), exist_ok=True)

    info_page = st.Page("info.py", title="Info")
    results_page = st.Page("results.py", title="Results")
    edges_page = st.Page("edges.py", title="Selected Edges")
    predictions_page = st.Page("predictions.py", title="Model Predictions")
    perms_page = st.Page("permutations.py", title="Permutations")

    pg = st.navigation([info_page, results_page, edges_page, predictions_page, perms_page])
    st.sidebar.markdown("""
# Confound-Corrected Connectome-Based Predictive Modeling

Python-Toolbox  
Software Version: 0.1.0  
Author: Nils R. Winter  
Github: https://github.com/wwu-mmll/cpm_python  

    """)

    pg.run()


if __name__ == '__main__':
    main()
