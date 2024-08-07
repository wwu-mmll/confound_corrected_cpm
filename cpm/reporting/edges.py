import streamlit as st
from plots.plots import corr_plot
from plots.cpm_chord_plot import plot_cpm_chord_plot


col1, col2 = st.columns([3, 1])

with col2:
    selected_metric = st.radio("Select a metric to display:", ["positive_edges", "negative_edges",
                                                                   "overlap_positive_edges", "overlap_negative_edges",
                                                                   "stability_positive_edges", "stability_negative_edges",
                                                                   "sig_stability_positive_edges",
                                                                   "sig_stability_negative_edges"])

with col1:
    tab1, tab2 = st.tabs(["Corr Plot", "Chord Plot"])
    with tab1:
        plot_name_corr = corr_plot(results_folder=st.session_state.results_directory, selected_metric=selected_metric)
        st.image(plot_name_corr)
    with tab2:
        plot_name_chord = plot_cpm_chord_plot(results_folder=st.session_state.results_directory, selected_metric=selected_metric)
        st.image(plot_name_chord, width=800)
