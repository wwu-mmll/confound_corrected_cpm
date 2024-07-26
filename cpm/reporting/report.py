import pandas as pd
import streamlit as st


preds = pd.read_csv('/home/nwinter/PycharmProjects/cpm_python/examples/tmp/macs_demo/predictions.csv')

st.title('Predictions')
st.header('My header')
st.dataframe(preds)

st.header('Scatter plot')
st.scatter_chart(data=preds, x='y_pred_full_both', y='y_true')

