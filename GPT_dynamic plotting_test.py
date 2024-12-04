import networkx as nx
import plotly.graph_objects as go

# Create a graph and add edges with attributes
G = nx.Graph()
G.add_edge(1, 2, weight=5, flow=2)
G.add_edge(2, 3, weight=3, flow=5)
G.add_edge(3, 4, weight=1, flow=10)

# Define a layout
pos = nx.spring_layout(G)

# Extract edge attributes and node positions
weights = [G[u][v]['weight'] for u, v in G.edges()]
flows = [G[u][v]['flow'] for u, v in G.edges()]

# Normalize weights and flows
max_weight = max(weights)
max_flow = max(flows)
edge_opacities = [weight / max_weight for weight in weights]  # Normalized for [0, 1]
edge_widths = [flow / max_flow * 5 for flow in flows]         # Scaled to desired range

# Prepare edge traces
edge_x = []
edge_y = []
edge_colors = []
edge_widths_scaled = []

for (u, v), opacity, width in zip(G.edges(), edge_opacities, edge_widths):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_colors.append(f'rgba(0, 0, 255, {opacity})')  # Use RGBA for color and opacity
    edge_widths_scaled.append(width)

# Create edge traces
edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=edge_widths_scaled[0], color=edge_colors[0]),
    hoverinfo='none',
    mode='lines'
)

# Adjust the edge attributes dynamically for all edges
edge_trace.line.aqua = edge_colors
edge_trace.line.width = edge_widths_scaled

# Prepare node traces
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    marker=dict(size=10, color='lightblue', line=dict(width=2, color='black')),
    text=list(G.nodes()),
    hoverinfo='text'
)

# Create the figure
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title="Network Graph with Edge Width and Opacity",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False)
)

fig.show()
