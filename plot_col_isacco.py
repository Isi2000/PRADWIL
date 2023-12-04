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
df = pd.read_json("data.json")
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

author_positions = {author: position for position, (author, _) in enumerate(borda, start=1)}
min_position = min(author_positions.values())
max_position = max(author_positions.values())

colormap = plt.cm.get_cmap("RdPu")

default_node_size = 1

node_sizes = {}
node_colors = {}

# Assign sizes and colors for authors in sub_borda
for author, position in author_positions.items():
    norm_pos = (position - min_position) / (max_position - min_position)
    norm_pos = 1-norm_pos
    c = colormap(norm_pos)
    node_colors[author] = c
    node_sizes[author] = default_node_size + norm_pos*2


nodes_color_list = []
nodes_color_size = []
for i in cc.nodes():
    nodes_color_list.append(node_colors[i])
    nodes_color_size.append(node_sizes[i])

#Loading positions
with open('positions.pkl', 'rb') as file:
    pos = pickle.load(file)

pos = nx.kamada_kawai_layout(cc, pos = pos)
labels = {auth: auth for auth, _ in borda[0:5]}
#plt.figure(figsize=(8,8))
# Draw the bipartite graph
nx.draw_networkx(cc, pos, node_color=nodes_color_list, node_size=nodes_color_size, labels =labels, font_size= 8) #width = edge_widths_list, edge_color = edge_colors_list)

plt.show()
