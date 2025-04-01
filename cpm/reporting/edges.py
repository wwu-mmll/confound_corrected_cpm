import streamlit as st

from plots.cpm_chord_plot import plot_netplotbrain


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


st.title('Brain Connections')

plots = []
edges = []
for metric in ["positive_edges", "negative_edges", "stability_positive_edges",
               "stability_negative_edges", "sig_stability_positive_edges", "sig_stability_negative_edges"]:
    plot_brainplot, edge_list = plot_netplotbrain(results_folder=st.session_state.results_directory,
                                                  selected_metric=metric)
    plots.append(plot_brainplot)
    edges.append(edge_list)


with st.container(border=0):
    st.markdown('### Selected Edges')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('Positive Edges')
        st.image(plots[0])
        st.download_button(
            label="Download edges",
            data=convert_df(edges[0]),
            file_name="pos_edges.csv",
            mime="text/csv",
        )
        #st.dataframe(edges[0])

    with col2:
        st.markdown('Negative Edges')
        st.image(plots[1])
        st.download_button(
            label="Download edges",
            data=convert_df(edges[1]),
            file_name="neg_edges.csv",
            mime="text/csv",
        )
        #st.dataframe(edges[1])

with st.container(border=0):
    st.markdown('### Edge Stability')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('Positive Edges')
        st.image(plots[2])
        st.download_button(
            label="Download edges",
            data=convert_df(edges[2]),
            file_name="pos_edges_stability.csv",
            mime="text/csv",
        )
        #st.dataframe(edges[2])

    with col2:
        st.markdown('Negative Edges')
        st.image(plots[3])
        st.download_button(
            label="Download edges",
            data=convert_df(edges[3]),
            file_name="neg_edges_stability.csv",
            mime="text/csv",
        )
        #st.dataframe(edges[3])

with st.container(border=0):
    st.markdown('### Edge Stability Significance')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('Positive Edges')
        st.image(plots[4])
        st.download_button(
            label="Download edges",
            data=convert_df(edges[4]),
            file_name="pos_edges_stability_significance.csv",
            mime="text/csv",
        )
        #st.dataframe(edges[4])

    with col2:
        st.markdown('Negative Edges')
        st.image(plots[5])
        st.download_button(
            label="Download edges",
            data=convert_df(edges[5]),
            file_name="neg_edges_stability_significance.csv",
            mime="text/csv",
        )
        #st.dataframe(edges[5])

