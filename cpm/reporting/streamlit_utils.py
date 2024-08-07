import os
import pandas as pd
import streamlit as st


def style_apa(df):
    styled_df = df.style.set_properties(
        **{
            'border': '1px solid black',
            'border-collapse': 'collapse',
            'padding': '5px 10px',
            'text-align': 'center'
        }
    ).set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('border', '1px solid black'),
                ('text-align', 'center'),
                ('font-weight', 'bold'),
                ('background-color', '#f9f9f9'),
                ('padding', '5px 10px'),
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('border', '1px solid black'),
                ('text-align', 'center'),
                ('padding', '5px 10px'),
            ]
        },
        {
            'selector': '.row_heading',
            'props': [
                ('text-align', 'left'),
                ('font-style', 'italic'),
                ('border-right', '1px solid black'),
                ('padding', '5px 10px'),
            ]
        },
        {
            'selector': '.index_name',
            'props': [
                ('text-align', 'left'),
                ('font-style', 'normal'),
                ('border-right', '1px solid black'),
                ('padding', '5px 10px'),
            ]
        }
    ]).format(precision=3)

    return styled_df


# Function to read CSV file from the given folder path
def load_data_from_folder(folder_path, filename):
    csv_path = os.path.join(folder_path, filename)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error(f"No CSV file found at path: {csv_path}")
        return None


def load_results_from_folder(folder_path, filename):
    csv_path = os.path.join(folder_path, filename)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, header=[0, 1], index_col=[0, 1])
    else:
        st.error(f"No CSV file found at path: {csv_path}")
        return None
