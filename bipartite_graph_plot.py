
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import analysis_functions as af
import random
import numpy as np
from matplotlib import cm
import matplotlib
# Load your data and create the bipartite graph
df = pd.read_json("./data/data.json")
G = af.create_bipartite_graph(df)

# Extract nodes
authors_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]
papers_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]

# Load borda data

borda = np.load("./results/borda.npy")
borda = [[author, int(position)] for author, position in borda]
impo_authors = [author[0] for author in borda[:5]]

# Create positions and colormap
author_positions = {author: position for position, (author, _) in enumerate(borda, start=1)}
pos_to_plot_grad = list(author_positions.values())
min_position = min(author_positions.values())
max_position = max(author_positions.values())
colormap = plt.cm.get_cmap("coolwarm")
#######################################
# Set default node size
default_node_size = 1

# Initialize node sizes and colors
node_sizes = {}
node_colors = {}

for author, position in author_positions.items():
    norm_pos = (position - min_position) / (max_position - min_position)
    norm_pos = 1-norm_pos

    if author in impo_authors:
        node_colors[author] = colormap(norm_pos)
        node_sizes[author] = default_node_size + norm_pos*100
    else:
        node_colors[author] = colormap(norm_pos)
        node_sizes[author] = default_node_size


# Assign default size and color for authors not in 
add_auth = [auth for auth in authors_nodes if auth not in author_positions.keys()]
for auth in add_auth:
    node_colors[auth] = colormap(0)
    node_sizes[auth] = default_node_size

for p in papers_nodes:
    node_colors[p] = 'black'
    node_sizes[p] = default_node_size

nodes_color_list = []
nodes_color_size = []
for i in G.nodes():
    nodes_color_list.append(node_colors[i])
    nodes_color_size.append(node_sizes[i])

#######################################
f = 0.1
edge_colors = {}
edge_widths = {}
edge_alpha = {}
# Iterate through edges and assign colors, widths, and alpha based on the colors of their associated authors
for edge in G.edges():
    author, paper = edge
    # Check if the authors are in author_positions
    if author in impo_authors:
        position = author_positions[author]
        # Normalize position
        normalized_position = (position - min(author_positions.values())) / (max(author_positions.values()) - min(author_positions.values()))
        normalized_position = 1 - normalized_position
        
        edge_colors[edge] = node_colors[author]
        edge_widths[edge] = normalized_position * f  # Adjust the multiplier as needed
        edge_alpha[edge] = 1 - normalized_position  # Adjust the alpha as needed (1.0 is fully opaque)
        
    elif paper in impo_authors:
        position = author_positions[paper]
        # Normalize position
        normalized_position = (position - min(author_positions.values())) / (max(author_positions.values()) - min(author_positions.values()))
        normalized_position = 1 - normalized_position 
        edge_colors[edge] = node_colors[paper]
        edge_widths[edge] = normalized_position * f  # Adjust the multiplier as needed
        edge_alpha[edge] = 1 - normalized_position  # Adjust the alpha as needed (1.0 is fully opaque)
    else:
        # Set a default color for edges not associated with colored authors
        edge_colors[edge] = colormap(0)
        edge_widths[edge] = 0.001 
        edge_alpha[edge] = 0.001  # Adjust

edge_colors_list = []
edge_widths_list = []
edge_alpha_list = []

for i in G.edges():
    edge_colors_list.append(edge_colors[i])
    edge_widths_list.append(edge_widths[i])
    edge_alpha_list.append(edge_alpha[i])




labels = {auth: auth for auth, _ in borda[:5]}

fig, ax = plt.subplots(figsize=(14, 10))
# Draw the bipartite graph
pos = nx.bipartite_layout(G, papers_nodes)
nx.draw_networkx(G, pos, node_color=nodes_color_list, node_size=nodes_color_size, labels =labels, font_size= 8, width = edge_widths_list, edge_color = edge_colors_list, font_color = 'black',  horizontalalignment='right')

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(borda))
fig.colorbar(cm.ScalarMappable(norm = norm, cmap=colormap), orientation = 'horizontal', label = "Authors colored with Borda ranking")

plt.text(0.32, 0.5, "Authors")
plt.text(-0.95, 0.5, "Papers")
plt.gca().invert_xaxis()

ax.set_facecolor('#E0E0E0')
plt.savefig('figura_bip.png')
