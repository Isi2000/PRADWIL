import networkx as nx
import community
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import analysis_functions as af
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

###
df = pd.read_json("./data/data.json")
G = af.create_bipartite_graph(df)
article_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
authors_nodes = set(G) - article_nodes
C = bipartite.weighted_projected_graph(G, authors_nodes) #weighted projection
number_of_connected_components = nx.number_connected_components(C)
connected_components = list(nx.connected_components(C))
filtered_connected_components = [comp for comp in connected_components if len(comp) > 1] 
lengths = [len(comp) for comp in filtered_connected_components]
sorted_lengths = sorted(lengths, reverse=True)
largest_cc = max(filtered_connected_components, key=len)
cc = C.subgraph(largest_cc).copy()
###
borda = np.load("./results/borda.npy")
borda = [[author, int(position)] for author, position in borda]

impo_authors = [author[0] for author in borda[:10]]
print(impo_authors)
author_positions = {author: position for position, (author, _) in enumerate(borda, start=1)}

min_position = min(author_positions.values())
max_position = max(author_positions.values())

colormap = plt.cm.get_cmap("coolwarm")
#colormap1 = plt.cm.get_cmap("Blues")

default_node_size = 0.01

node_sizes = {}
node_colors = {}


# Assign sizes and colors for authors in sub_borda
for author, position in author_positions.items():
    norm_pos = (position - min_position) / (max_position - min_position)
    norm_pos = 1-norm_pos

    if author in impo_authors:
        node_colors[author] = colormap(norm_pos)
        node_sizes[author] = default_node_size + norm_pos*300
    else:
        node_colors[author] = colormap(norm_pos)
        node_sizes[author] = default_node_size


nodes_color_list = []
nodes_color_size = []
for i in cc.nodes():
    nodes_color_list.append(node_colors[i])
    nodes_color_size.append(node_sizes[i])

    
edge_colors = {}
edge_widths = {}
edge_alpha = {}
# Iterate through edges and assign colors, widths, and alpha based on the colors of their associated authors
for edge in cc.edges():
    author, author1 = edge
    position = min(author_positions[author], author_positions[author1])
 
    c_author = list(author_positions.keys())[list(author_positions.values()).index(position)]
    
    norm_pos = (position - min_position) / (max_position - min_position)
    norm_pos = 1-norm_pos
    edge_colors[edge] = node_colors[c_author]
    edge_widths[edge] = norm_pos  * 0.01 
    edge_alpha[edge] = 1.0-norm_pos  
edge_colors_list = []
edge_widths_list = []
edge_alpha_list = []

for i in cc.edges():
    edge_colors_list.append(edge_colors[i])
    edge_widths_list.append(edge_widths[i])
    edge_alpha_list.append(edge_alpha[i])


print("done")
#Loading positions
with open('positions1.pkl', 'rb') as file:
    pos = pickle.load(file)


labels = {auth: auth for auth, _ in borda[0:10]}
#plt.figure(figsize=(8,8))
# Draw the bipartite graph
nx.draw_networkx(cc, pos, node_color=nodes_color_list, node_size=nodes_color_size, labels =labels, font_size= 8, width = edge_widths_list, edge_color = edge_colors_list, font_color = 'yellow')

plt.show()
