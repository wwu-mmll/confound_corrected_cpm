import streamlit as st
from plots.plots import scatter_plot


plot_name = scatter_plot(st.session_state.df_predictions, st.session_state.results_directory)
st.image(plot_name, width=1000)
