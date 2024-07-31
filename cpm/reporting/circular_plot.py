import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from bokeh.io import output_file, save, show
from bokeh.models import LabelSet, ColumnDataSource
from bokeh.palettes import Category20

# Enable bokeh extension in holoviews
hv.extension('bokeh')

# Step 1: Generate a symmetric data matrix
np.random.seed(0)
num_nodes = 20  # Number of nodes
data_matrix = np.random.rand(num_nodes, num_nodes)
data_matrix = (data_matrix + data_matrix.T) / 2  # Make it symmetric
np.fill_diagonal(data_matrix, 0)  # No self-connections

# Step 2: Create node labels and network labels
node_labels = [f'Region {i+1}' for i in range(num_nodes)]
network_labels = ['Network 1'] * 4 + ['Network 2'] * 4 + ['Network 3'] * 4 + \
                 ['Network 4'] * 4 + ['Network 5'] * 4

# Step 3: Create a DataFrame for chord diagram
df = pd.DataFrame(data_matrix, columns=node_labels, index=node_labels)
df = df.stack().reset_index()
df.columns = ['source', 'target', 'value']
df = df[df['value'] > 0.5]  # Filter for stronger connections

# Step 4: Assign network colors
unique_networks = list(set(network_labels))
network_colors = dict(zip(unique_networks, Category20[len(unique_networks)]))
node_color_map = [network_colors[network_labels[i]] for i in range(num_nodes)]

# Step 5: Create the Chord Diagram
nodes_df = pd.DataFrame({'index': node_labels, 'network': network_labels, 'color': node_color_map})
nodes = hv.Dataset(nodes_df, 'index')
chord = hv.Chord((df, nodes)).opts(
    opts.Chord(
        cmap='Category20',
        edge_cmap='coolwarm',
        edge_color='value',
        edge_alpha=0.8,
        node_color='color',
        node_size=15,
        labels='index',  # Ensure labels correspond to node index
        label_text_font_size='10pt',
        width=800,  # Adjust width
        height=800,  # Adjust height
    )
)

# Step 6: Convert to Bokeh plot and Extract Node Positions
bokeh_plot = hv.render(chord, backend='bokeh')

# Extract node positions from the layout
node_positions = bokeh_plot.renderers[1].layout_provider.graph_layout
node_positions_df = pd.DataFrame(node_positions).T
node_positions_df.columns = ['x', 'y']
node_positions_df['labels'] = node_labels

# Add permanent labels to the nodes
label_source = ColumnDataSource(node_positions_df)
labels = LabelSet(x='x', y='y', text='labels', source=label_source, text_font_size='10pt',
                  text_align='center', text_baseline='middle')

bokeh_plot.add_layout(labels)

# Render and save the plot
output_file('chord_plot.html')  # Define output file
save(bokeh_plot)  # Save plot to HTML file
show(bokeh_plot)  # Display the plot
