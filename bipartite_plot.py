import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import analysis_functions as af
import random
import numpy as np

# Load your data and create the bipartite graph
df = pd.read_json("data.json")
G = af.create_bipartite_graph(df)

# Extract nodes
authors_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]
papers_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]

# Load borda data
borda = np.load("./results/borda.npy")
borda = [[author, int(position)] for author, position in borda]
sub_borda = borda[:6]
sub_borda = sub_borda[::-1] #altrimenti devo riscrivere tutto
print(sub_borda)
# Create positions and colormap
sub_author_positions = {author: position for position, (author, _) in enumerate(sub_borda, start=1)}
sub_min_position = min(sub_author_positions.values())
sub_max_position = max(sub_author_positions.values())
colormap = plt.cm.get_cmap("RdPu")

#######################################
# Set default node size
default_node_size = 1

# Initialize node sizes and colors
node_sizes = {}
node_colors = {}

# Assign sizes and colors for authors in sub_borda
for author, position in sub_author_positions.items():
    norm_pos = (position - sub_min_position) / (sub_max_position - sub_min_position)
    norm_pos = norm_pos - (1*0.2) + 0.2
    c = colormap(norm_pos)
    node_colors[author] = c
    node_sizes[author] = default_node_size + norm_pos * 100

# Assign default size and color for authors not in sub_borda
add_auth = [auth for auth in authors_nodes if auth not in sub_author_positions.keys()]
for auth in add_auth:
    node_colors[auth] = 'lightblue'
    node_sizes[auth] = default_node_size

for p in papers_nodes:
    node_colors[p] = 'lightblue'
    node_sizes[p] = default_node_size

nodes_color_list = []
nodes_color_size = []
for i in G.nodes():
    nodes_color_list.append(node_colors[i])
    nodes_color_size.append(node_sizes[i])
#######################################
edge_colors = {}
edge_widths = {}
edge_alpha = {}
# Iterate through edges and assign colors, widths, and alpha based on the colors of their associated authors
for edge in G.edges():
    author, paper = edge
    # Check if the authors are in sub_author_positions
    if author in dict(sub_author_positions).keys():
        position = sub_author_positions[author]
        # Normalize position
        normalized_position = (position - min(sub_author_positions.values())) / (max(sub_author_positions.values()) - min(sub_author_positions.values()))
        # Assign the color of the first author to the edge
        edge_colors[edge] = node_colors[author]
        # Assign edge width based on normalized author's position
        edge_widths[edge] = normalized_position  # Adjust the multiplier as needed
        # Assign alpha based on normalized author's position
        edge_alpha[edge] = 1.0-normalized_position  # Adjust the alpha as needed (1.0 is fully opaque)
        
    elif paper in dict(sub_author_positions).keys():
        position = sub_author_positions[paper]
        # Normalize position
        normalized_position = (position - min(sub_author_positions.values())) / (max(sub_author_positions.values()) - min(sub_author_positions.values()))
        normalized_position = normalized_position * (1 - 0.2) + 0.2
        edge_colors[edge] = node_colors[paper]
        # Assign edge width based on normalized paper's position
        edge_widths[edge] = 1 * normalized_position  # Adjust the multiplier as needed
        # Assign alpha based on normalized paper's position
        edge_alpha[edge] = 1 - normalized_position  # Adjust the alpha as needed (1.0 is fully opaque)
    else:
        # Set a default color for edges not associated with colored authors
        edge_colors[edge] = 'snow'
        edge_widths[edge] = 0.01
        edge_alpha[edge] = 0.01  # Adjust

edge_colors_list = []
edge_widths_list = []
edge_alpha_list = []

for i in G.edges():
    edge_colors_list.append(edge_colors[i])
    edge_widths_list.append(edge_widths[i])
    edge_alpha_list.append(edge_alpha[i])



labels = {auth: auth for auth, _ in sub_borda[1:]}
#plt.figure(figsize=(8,8))
# Draw the bipartite graph
pos = nx.bipartite_layout(G, papers_nodes)
nx.draw_networkx(G, pos, node_color=nodes_color_list, node_size=nodes_color_size, labels =labels, font_size= 8, width = edge_widths_list, edge_color = edge_colors_list)

plt.show()
