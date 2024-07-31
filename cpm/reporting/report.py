import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from glob import glob
import numpy as np


dirname = os.path.join('/home/nwinter/PycharmProjects/cpm_python/cpm/reporting/fonts/*')
font_folders = glob(dirname)
for folder in font_folders:
    for font in fm.findSystemFonts(folder):
        fm.fontManager.addfont(font)


# Path to the downloaded font file (update this to the actual path of your .ttf or .otf file)
#font_path = '/home/nwinter/PycharmProjects/cpm_python/cpm/reporting/fonts/ShareTech/ShareTech-Regular.ttf'  # Example: './fonts/Roboto-Regular.ttf'
font_path = '/home/nwinter/PycharmProjects/cpm_python/cpm/reporting/fonts/Lato/Lato-Regular.ttf'  # Example: './fonts/Roboto-Regular.ttf'

# Create a font object from the font file
# Register the font
roboto_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = roboto_font.get_name()

plt.rcParams['font.size'] = 10

# Set up the page configuration
st.set_page_config(layout="wide")

# Hardcoded default folder path for debugging
DEFAULT_FOLDER_PATH = '/home/nwinter/PycharmProjects/cpm_python/examples/tmp/example_simulated_data/'


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


# Sample Pandas styling function
def apply_table_styles(df):
    return df.style.format(precision=3).set_table_styles([{
        'selector': 'thead th',
        'props': [('background-color', '#f2f2f2'), ('font-weight', 'bold'), ('font-size', '110%'), ('color', '#333333')]
    }, {
        'selector': 'thead tr th:first-child',
        'props': [('display', 'none')]  # Remove index header
    }, {
        'selector': 'tbody tr th',
        'props': [('display', 'none')]  # Remove row indices
    }, {
        'selector': 'tbody td',
        'props': [('background-color', '#ffffff'), ('border', '1px solid #dddddd')]
    }, {
        'selector': 'tbody tr:nth-child(even) td',
        'props': [('background-color', '#f9f9f9')]
    }])


# Function to display the dataframe
def display_table(df):
    st.write(style_apa(df).to_html(), unsafe_allow_html=True)


# Function to create scatter plots of predictions
def scatter_plot(df, results_folder):
    g = sns.FacetGrid(df, row="network", col="model", margin_titles=True)
    g.map(sns.regplot, "y_true", "y_pred", scatter_kws={"alpha": 0.7, "s": 10, "edgecolor": "white"})
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=14)
    plot_name = os.path.join(results_folder, "plots", 'predictions.png')
    g.fig.savefig(plot_name)
    st.image(plot_name, width=800)

def vector_to_matrix_3d(vector_2d, shape):
    """
    Convert a vector containing strictly upper triangular parts back to a 3D matrix.

    Parameters:
    vector_2d (np.ndarray): A 2D array where each row is a vector of the strictly upper triangular part of a 2D matrix.
    shape (tuple): The shape of the original 3D matrix, (n_samples, n, n).

    Returns:
    np.ndarray: The reconstructed 3D matrix of shape (n_samples, n, n).
    """
    n_samples, n, _ = shape
    # Create an empty 3D matrix to fill
    matrix_3d = np.zeros((n_samples, n, n))

    # Create an index matrix for the strictly upper triangular indices
    row_indices, col_indices = np.tril_indices(n, k=-1)  # k=1 excludes the diagonal
    upper_tri_indices = np.ravel_multi_index((row_indices, col_indices), (n, n))

    # Flatten the 3D matrix along the last two dimensions
    flat_matrix = matrix_3d.reshape(n_samples, -1)

    # Place the strictly upper triangular elements into the corresponding positions
    np.put_along_axis(flat_matrix, upper_tri_indices[None, :], vector_2d, axis=1)

    return matrix_3d

def corr_plot(results_folder):
    selected_metric = st.selectbox("Select a metric to display:", ["positive_edges", "negative_edges",
                                                                    "overlap_positive_edges", "overlap_negative_edges",
                                                                   "weights_positive_edges", "weights_negative_edges"])

    corr = np.load(os.path.join(results_folder, f"{selected_metric}.npy"))

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr.T, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plot_name = os.path.join(results_folder, "plots", 'corr.png')
    fig.savefig(plot_name)
    st.image(plot_name, width=800)


def bar_plot(df, selected_metric, results_folder):

    # Dropdown for metric selection
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left plot for connectome, covariates, and full
    sns.pointplot(
        data=df[df['model'].isin(['connectome', 'covariates', 'full'])],
        x="network", y=selected_metric, hue="model",
        dodge=0.4, linestyle="none", errorbar="sd", capsize=.1,
        ax=axes[0]
    )

    # Right plot for increment
    sns.pointplot(
        data=df[df['model'].isin(['increment'])],
        x="network", y=selected_metric, hue="model",
        linestyle="none", errorbar="sd", capsize=.1,
        ax=axes[1]
    )
    plot_name = os.path.join(results_folder, "plots", 'point_plot.png')
    fig.savefig(plot_name, dpi=300)
    st.markdown(f'## {selected_metric}')
    st.image(plot_name, width=1000)
    #st.pyplot(fig)

# Function to calculate p-values
def calculate_p_values(true_results, perms):
    grouped_true = true_results.groupby(['network', 'model'])
    grouped_perms = perms.groupby(['network', 'model'])

    p_values = []

    for (name, true_group), (_, perms_group) in zip(grouped_true, grouped_perms):
        p_value_series = _calculate_group_p_value(true_group, perms_group)
        p_values.append(pd.DataFrame(p_value_series).T.assign(network=name[0], model=name[1]))

    p_values_df = pd.concat(p_values).reset_index(drop=True)
    p_values_df = p_values_df.set_index(['network', 'model'])

    return p_values_df

def _calculate_group_p_value(true_group, perms_group):
    result_dict = {}

    for column in true_group.columns:
        condition_count = 0
        if column.endswith('error'):
            condition_count = (true_group[column].values[0] > perms_group[column]).sum()
        elif column.endswith('score'):
            condition_count = (true_group[column].values[0] < perms_group[column]).sum()

        result_dict[column] = condition_count / (len(perms_group[column]) + 1)

    return pd.Series(result_dict)

# Permutations page function
def permutations_page(original_results_directory, selected_metric):

    true_results = pd.read_csv(os.path.join(original_results_directory, 'cv_results_mean_std.csv'), header=[0, 1],
                               index_col=[0, 1])
    true_results = true_results.loc[:, true_results.columns.get_level_values(1) == 'mean']
    true_results.columns = true_results.columns.droplevel(1)

    perm_results = []
    for i in range(50):  # Adjust the range as needed
        res_dir = os.path.join(original_results_directory, 'permutation', f'{i}')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        res = pd.read_csv(os.path.join(res_dir, 'cv_results_mean_std.csv'), header=[0, 1], index_col=[0, 1])
        res = res.loc[:, res.columns.get_level_values(1) == 'mean']
        res.columns = res.columns.droplevel(1)
        res['permutation'] = i
        res = res.set_index('permutation', append=True)

        perm_results.append(res)

    concatenated_df = pd.concat(perm_results)

    p_values = calculate_p_values(true_results, concatenated_df)
    p_values.to_csv(os.path.join(original_results_directory, 'p_values.csv'))
    # Style the DataFrame
    styled_df = style_apa(p_values)

    # Display in Streamlit
    st.write(styled_df.to_html(), unsafe_allow_html=True)

    long_perms = concatenated_df.reset_index().melt(id_vars=['network', 'model'], var_name='metric',
                                                    value_name='permuted_value')
    true_melted = true_results.reset_index().melt(id_vars=['network', 'model'], var_name='metric',
                                                  value_name='true_value')
    merged = pd.merge(long_perms, true_melted, on=['network', 'model', 'metric'])

    # Filter data based on selected metric
    metric_data = merged[merged['metric'] == selected_metric]

    g = sns.FacetGrid(metric_data, row="network", col="model", margin_titles=True, sharex=False, sharey=False, height=2,
                      aspect=1.5)

    # Define a plotting function that handles each subset
    def plot_histogram_with_line(data, **kwargs):
        sns.histplot(data['permuted_value'], kde=False, **kwargs)
        for true_value in data['true_value'].unique():
            plt.axvline(true_value, color='red', linestyle='dashed', linewidth=1)

    # Map the plotting function to each subset
    g.map_dataframe(plot_histogram_with_line)

    # Adjust plot titles and labels
    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=14)
    g.set_axis_labels(f"{selected_metric}", 'Count')

    # Display the plot
    plot_name = os.path.join(original_results_directory, "plots", 'permutations.png')
    g.fig.savefig(plot_name, dpi=300)
    st.markdown(f'## {selected_metric}')
    st.image(plot_name, width=1000)
    #st.pyplot(g.fig)

# Main part of the app
def main():
    st.title('Connectome-Based Predictive Modeling')
    st.sidebar.header('Analysis')
    st.sidebar.markdown('Here is a description of the analysis Here is a description of the analysis')

    folder_path = st.sidebar.text_input("Enter the folder path containing the CSV files:", "")
    if not folder_path:
        folder_path = DEFAULT_FOLDER_PATH

    if os.path.exists(folder_path):
        os.makedirs(os.path.join(folder_path, "plots"), exist_ok=True)
        df = pd.read_csv(os.path.join(folder_path, 'cv_results.csv'))
        selected_metric = st.sidebar.selectbox("Select a metric to display:", list(df.columns)[4:], key="bar_metric")

        df_main = load_results_from_folder(folder_path, 'cv_results_mean_std.csv')
        df_scatter = load_data_from_folder(folder_path, 'predictions.csv')

        tab1, tab2, tab3, tab4 = st.tabs(["Results", "Predictions", "Edges", "Permutations"])

        with tab1:
            st.title('Main Results')
            bar_plot(df, selected_metric, folder_path)
            display_table(df_main)

        with tab2:
            scatter_plot(df_scatter, folder_path)

        with tab3:
            corr_plot(results_folder=folder_path)

        with tab4:
            st.title('Permutation Test Results')
            permutations_page(folder_path, selected_metric)
    else:
        st.sidebar.error('The provided folder path does not exist.')

if __name__ == '__main__':
    main()