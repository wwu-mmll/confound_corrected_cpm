import streamlit as st
from plots.plots import bar_plot, bar_plot_plotly
from streamlit_utils import style_apa

st.title('Main Results')

tab1, tab2 = st.tabs(["Error Plot", "Table"])

with tab2:
    with st.container():
        st.html(style_apa(st.session_state.df_main).to_html())

with tab1:
    col1, col2 = st.columns([3, 1])

    with col2:
        with st.container(border=False):
            selected_metric = st.radio("Select a metric to display:", list(st.session_state.df.columns)[3:-1])

    with col1:
        plot_name, fig = bar_plot(st.session_state.df, selected_metric, st.session_state.results_directory)
        #fig = bar_plot_plotly(st.session_state.df, selected_metric, st.session_state.results_directory)

        with st.container(border=False):
            #st.plotly_chart(fig, use_container_width=True)
            st.image(plot_name)
