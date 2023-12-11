import networkx as nx
import community
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import pandas as pd
import analysis_functions as af
from collections import defaultdict
import numpy as np

# Read the data-------------------------------------------------------------

df = pd.read_json("./data/data.json")

#Building the bipartite graph------------------------------------------------

print('Creating the author-paper bipartite graph...')

G = af.create_bipartite_graph(df)

print('Done!')

# Coauthorship graph---------------------------------------------------------

print('Projecting the graph on the authors nodes (collaboration network)... ')

article_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
authors_nodes = set(G) - article_nodes

C = bipartite.weighted_projected_graph(G, authors_nodes) #weighted projection
connected_components = list(nx.connected_components(C))

# Filtering out the single nodes
filtered_connected_components = [comp for comp in connected_components if len(comp) > 1] 

largest_cc = max(filtered_connected_components, key=len)
print('Number of nodes of the largest connected component:', len(largest_cc))

#Building a network of the largest connected component-----------------------

cc = C.subgraph(largest_cc).copy() 

# Louvain communities---------------------------------------------------------




partition = community.best_partition(cc, weight='weight', randomize = False, random_state=42)
print(partition)
print('Number of communities:', len(set(partition.values())))
print('Modularity:', community.modularity(partition, cc))

# Creare un dizionario di liste dove le chiavi sono gli identificatori delle comunità
community_lists = defaultdict(list)
print(community_lists)
for node, community_id in partition.items():
    community_lists[community_id].append(node)


    list_of_communities = list(community_lists.values())
list_of_communities.sort(key=len, reverse=True)
communities_to_plot = [list_of_communities[i] for i in range(9)]

subgraphs = []
for i, community in enumerate(communities_to_plot):
    community_subgraph = cc.subgraph(community).copy()
    subgraphs.append(community_subgraph)

print(communities_to_plot)
###diocane here we go again
borda = np.load("./results/borda.npy")
eig = np.load("./results/eigenvector_centrality.npy")
deg = np.load("./results/degree_centrality.npy")
bet = np.load("./results/betweeness_centrality.npy")
clo = np.load("./results/closeness_centrality.npy")

eig_positions = {author: position for position, (author, _) in enumerate(eig, start=1)}
deg_positions = {author: position for position, (author, _) in enumerate(deg, start=1)}
bet_positions = {author: position for position, (author, _) in enumerate(bet, start=1)}
clo_positions = {author: position for position, (author, _) in enumerate(clo, start=1)}
# Assuming eig_positions, deg_positions, bet_positions, clo_positions are already defined

# Combine all centrality measures into a list of dictionaries
all_positions = [eig_positions, deg_positions, bet_positions, clo_positions]

# Create a dictionary to store the lowest positions for each author
lowest_positions = {}

# Iterate over authors in the first centrality measure
for author in eig_positions.keys():
    positions = [pos[author] for pos in all_positions]
    min_position = min(positions)
    
    # Check for ties
    tied_measures = [name for name, pos in zip(["eig", "deg", "bet", "clo"], positions) if pos == min_position]
    
    # Store the result in the dictionary
    if len(tied_measures) == 1:
        lowest_positions[author] = tied_measures[0]
    else:
        lowest_positions[author] = "tied"

color_m = {
    'eig': 'red',
    'deg': 'green',
    'bet': 'blue',
    'clo': 'yellow',
    'tied': 'purple', 
    # Add more combinations as needed
}


authors_and_colors = {author: color_m[position] for author, position in lowest_positions.items()}




borda = [[author, int(position)] for author, position in borda]

author_positions = {author: position for position, (author, _) in enumerate(borda, start=1)}
pos_to_plot_grad = list(author_positions.values())
min_position = min(author_positions.values())
max_position = max(author_positions.values())

default_node_size = 1

# Initialize node sizes and colors
node_sizes = {}
node_colors = {}
auth_norm_pos = {}
for author, position in author_positions.items():
    norm_pos = (position - min_position) / (max_position - min_position)
    norm_pos = 1-norm_pos
    auth_norm_pos[author] = norm_pos

n_colors = []
n_size = []
for comm in subgraphs:
    cols = []
    size = []
    for author in comm.nodes():
        if author in auth_norm_pos.keys():
            cols.append(authors_and_colors[author])
            size.append(auth_norm_pos[author])
    n_colors.append(cols)
    n_size.append(size)


    
from matplotlib import gridspec
import matplotlib as mpl

from matplotlib import gridspec
import matplotlib as mpl
# Set up the subplot grid
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)

# Loop through the subgraphs and add them to the subplot grid
for i, subgraph in enumerate(subgraphs):
    ax = plt.subplot(gs[i])

    # Extract colors and size data for nodes in the subgraph
    cols = n_colors[i]
    size = n_size[i]
    pos = nx.kamada_kawai_layout(subgraph)
    # Draw the subgraph
    nx.draw(subgraph, pos, node_color=cols, node_size=size, with_labels=False, ax=ax, width = 0.1)

    ax.set_title(f"Community rank: {i + 1}")

plt.savefig('color_comm.png')
