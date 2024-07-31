# Load d3blocks
from d3blocks import D3Blocks
#
# Initialize
d3 = D3Blocks(chart='Chord', frame=False)
#
# Import example
df = d3.import_example('energy')
#
# Node properties
d3.set_node_properties(df, opacity=0.2, cmap='tab20')
d3.set_edge_properties(df, color='source', opacity='source')
#
# Show the chart
d3.show()
#
# Make some edits to highlight the Nuclear node
# d3.node_properties
d3.node_properties.get('Nuclear')['color']='#ff0000'
d3.node_properties.get('Nuclear')['opacity']=1
# Show the chart
#
d3.show()
# Make edits to highlight the Nuclear Edge
d3.edge_properties.loc[(d3.edge_properties['source'] == 'Nuclear') & (d3.edge_properties['target'] == 'Thermal generation'), 'color'] = '#ff0000'
d3.edge_properties.loc[(d3.edge_properties['source'] == 'Nuclear') & (d3.edge_properties['target'] == 'Thermal generation'), 'opacity'] = 0.8
d3.edge_properties.loc[(d3.edge_properties['source'] == 'Nuclear') & (d3.edge_properties['target'] == 'Thermal generation'), 'weight'] = 1000
#
# Show the chart
d3.show()