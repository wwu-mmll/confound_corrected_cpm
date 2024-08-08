import streamlit as st
from plots.plots import scatter_plot

st.title('Predictions')

plot_name = scatter_plot(st.session_state.df_predictions, st.session_state.results_directory)
with st.container(border=True):
    st.image(plot_name, width=800)
