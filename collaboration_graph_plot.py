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

# Read the data-------------------------------------------------------------

df = pd.read_json("data.json")

#Building the bipartite graph------------------------------------------------

print('Creating the author-paper bipartite graph...')

G = af.create_bipartite_graph(df)

print('Done!')

# Coauthorship graph---------------------------------------------------------

print('Projecting the graph on the authors nodes (collaboration network)... ')

article_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
authors_nodes = set(G) - article_nodes

C = bipartite.weighted_projected_graph(G, authors_nodes) #weighted projection

print('Done!')

# Analysis of the graph components-------------------------------------------

number_of_connected_components = nx.number_connected_components(C)
connected_components = list(nx.connected_components(C))

# Filtering out the single nodes
filtered_connected_components = [comp for comp in connected_components if len(comp) > 1] 

lengths = [len(comp) for comp in filtered_connected_components]
sorted_lengths = sorted(lengths, reverse=True)

largest_cc = max(filtered_connected_components, key=len)
print('Number of nodes of the largest connected component:', len(largest_cc))

#Building a network of the largest connected component-----------------------

cc = C.subgraph(largest_cc).copy()

# Louvain communities---------------------------------------------------------

partition = community.best_partition(cc, weight='weight')

print(len(list(partition.values())))
# Generare il grafico con tutte le comunità colorate in modo diverso 

# Estrai il colore dei nodi basato sulle partizioni
node_colors = np.array(list(partition.values()))

# Estrai il peso degli archi
edge_weights = [cc[u][v]['weight'] for u, v in cc.edges()]

# Imposta la grandezza dei nodi proporzionale al loro grado
node_size = [deg*0.2 for _, deg in cc.degree()]

# Imposta lo spessore degli archi proporzionale al loro peso
edge_width = [weight*0.1 for weight in edge_weights]

# Calcola il grado di ciascun nodo e ordina i nodi per grado in modo decrescente
nodes_by_degree = sorted(cc.degree, key=lambda x: x[1], reverse=True)

# Estrai solo i primi 10 nodi
top_10_nodes = [node for node, _ in nodes_by_degree[:10]]

# Plot of the collaboration network------------------------------------------

#Loading positions
with open('positions_3d.pkl', 'rb') as file:
    pos = pickle.load(file)

# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in sorted(cc)])
print(len(node_xyz))
edge_xyz = np.array([(pos[u], pos[v]) for u, v in cc.edges()])

# Create the 3D figure
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")

# Estrai il peso degli archi
edge_weights = [cc[u][v]['weight'] for u, v in cc.edges()]

# Imposta lo spessore degli archi proporzionale al loro peso
edge_width = [weight*0.1 for weight in edge_weights]

# Converti i colori dei nodi in un array NumPy di tipo float
#node_colors = np.array(node_colors).astype(float)
print(len(node_colors))
print(node_colors)
node_colors = np.array(node_colors, dtype='float64')

# Definisci la colormap (puoi sceglierne una tra quelle disponibili)
cmap = cm.get_cmap('viridis')  # 'viridis' è solo un esempio, puoi cambiarlo

# Normalizza i valori float tra 0 e 1 (richiesto dalla colormap)
normalize = plt.Normalize(min(node_colors), max(node_colors))

# Mappa i valori float alla colormap
colors = [cmap(normalize(value)) for value in node_colors]

# Ora 'colors' contiene i colori associati ai valori float
print(colors)

ax.scatter(*node_xyz.T, s = 3, c = colors, linewidth = 0, cmap='viridis')

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray", linewidth=0.01)

def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

_format_axes(ax)
fig.tight_layout()
plt.show()

#Saving the figure
#print('Saving the figure...')
#import os
#os.makedirs('images', exist_ok=True)
#plt.savefig('images/3d_collaboration_network.png', dpi = 300)
#print('Saved!')



"""

#PER PLOTTARE GRAFO COMUNITA'
# Generare il dendrogramma
dendrogram_data = community.generate_dendrogram(cc, weight='weight')

for level in range(len(dendrogram_data)):
    partition_at_level = community.partition_at_level(dendrogram_data, level)
    print('number of communities at level', level, 'is', len(set(partition_at_level.values())))
    print("modularity at level", level, "is",
          community.modularity(partition_at_level, cc))
    
induced_graph = community.induced_graph(partition, cc)
nx.draw(induced_graph, node_size=50, with_labels=False)
plt.show()


#PER PLOTTARE MAIN COMMUNITY
from collections import defaultdict

# Supponendo che 'partition' sia il risultato di community.best_partition(cc)
partition = community.best_partition(cc, weight='weight')

# Creare un dizionario di liste dove le chiavi sono gli identificatori delle comunità
community_lists = defaultdict(list)

for node, community_id in partition.items():
    community_lists[community_id].append(node)

# Ora 'community_lists' è un dizionario in cui le chiavi sono gli identificatori delle comunità
# e i valori sono le liste di nodi appartenenti a ciascuna comunità.

# Convertire il dizionario in una lista di liste
list_of_communities = list(community_lists.values())

list_of_communities.sort(key=len, reverse=True)

main_community = list_of_communities[0]

# Estrai il sottografo corrispondente alla main community
main_community_subgraph = cc.subgraph(main_community).copy()

# Imposta il layout del grafico
pos = nx.spring_layout(main_community_subgraph)

# Estrai il peso degli archi
edge_weights = [main_community_subgraph[u][v]['weight'] for u, v in main_community_subgraph.edges()]

# Imposta la grandezza dei nodi proporzionale al loro grado
node_size = [deg * 10 for _, deg in main_community_subgraph.degree()]

# Imposta lo spessore degli archi proporzionale al loro peso
edge_width = [weight * 2 for weight in edge_weights]

# Imposta la grandezza dei label
label_font_size = 6

# Disegna il grafico
nx.draw(main_community_subgraph, pos, with_labels=False, font_size=label_font_size,
        node_size=node_size, width=edge_width, font_color='black', alpha=0.7)

# Visualizza il grafico
plt.show()
"""