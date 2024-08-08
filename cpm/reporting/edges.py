import streamlit as st
from plots.plots import corr_plot
from plots.cpm_chord_plot import plot_cpm_chord_plot


st.title('Selected Edges')

col1, col2 = st.columns([3, 1])

with col2:
    with st.container(border=1):
        selected_metric = st.radio("Select a metric to display:", ["positive_edges", "negative_edges",
                                                                   "stability_positive_edges", "stability_negative_edges",
                                                                   "sig_stability_positive_edges",
                                                                   "sig_stability_negative_edges"])

plot_name_corr = corr_plot(results_folder=st.session_state.results_directory, selected_metric=selected_metric)
plot_name_chord, plot_glass_brain = plot_cpm_chord_plot(results_folder=st.session_state.results_directory,
                                                        selected_metric=selected_metric)

with col1:
    t2col1, t2col2 = st.columns([1, 1])
    with t2col1:
        with st.container(border=1):
            st.image(plot_name_corr)
            st.markdown('Figure 1: Heatmap of Edges')

    with t2col2:
        with st.container(border=1):
            st.image(plot_name_chord)
            st.markdown('Figure 2: Chord Plot')
    with st.container(border=1):
        st.image(plot_glass_brain)
        st.markdown('Figure 3: Glass Brain')


