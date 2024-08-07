import streamlit as st
from plots.plots import bar_plot
from streamlit_utils import style_apa

st.title('Main Results')

col1, col2 = st.columns([3, 1])

with col2:
    with st.container(border=1):
        selected_metric = st.radio("Select a metric to display:", list(st.session_state.df.columns)[3:-1])

with col1:
    tab1, tab2 = st.tabs(["Error Plot", "Table"])

    with tab1:
        plot_name, fig = bar_plot(st.session_state.df, selected_metric, st.session_state.results_directory)
        st.image(plot_name)
        #st.pyplot(fig)
    with tab2:
        st.html(style_apa(st.session_state.df_main).to_html())
