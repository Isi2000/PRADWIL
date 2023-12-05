import networkx as nx
import community
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import pandas as pd
import analysis_functions as af
from collections import defaultdict

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

print('Done!')

# Analysis of the graph components-------------------------------------------

connected_components = list(nx.connected_components(C))

# Filtering out the single nodes
filtered_connected_components = \
    [comp for comp in connected_components if len(comp) > 1] 


# Computing the length of all the connected components

largest_cc = max(filtered_connected_components, key=len)
print('Number of nodes of the largest connected component:', len(largest_cc))

#Building a network of the largest connected component-----------------------

cc = C.subgraph(largest_cc).copy() 

# Louvain communities---------------------------------------------------------

partition = community.best_partition(cc, weight='weight', randomize = False, random_state=42)
print('Number of communities:', len(set(partition.values())))
print('Modularity:', community.modularity(partition, cc))

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
print('Number of nodes of the main community:', len(main_community))
print('Percentage of nodes of the main community:', len(main_community) / len(cc))

# Estrai il sottografo corrispondente alla main community
main_community_subgraph = cc.subgraph(main_community).copy()

# Imposta il layout del grafico
pos = nx.kamada_kawai_layout(main_community_subgraph)

# Estrai il peso degli archi
edge_weights = [main_community_subgraph[u][v]['weight'] for u, v in main_community_subgraph.edges()]

# Imposta la grandezza dei nodi proporzionale al loro grado
node_size = [deg * 0.1 for _, deg in main_community_subgraph.degree()]

# Imposta lo spessore degli archi proporzionale al loro peso
edge_width = [weight * 0.1 for weight in edge_weights]

# Imposta la grandezza dei label
label_font_size = 6

# Disegna il grafico
nx.draw(main_community_subgraph, pos, with_labels=False, font_size=label_font_size,
        node_size=node_size, width=edge_width, font_color='black', alpha=0.7)

# Visualizza il grafico
plt.show()

# Plotting the communities---------------------------------------------------
    
induced_graph = community.induced_graph(partition, cc)
pos = nx.kamada_kawai_layout(induced_graph)
nx.draw(induced_graph, pos = pos, node_size=10, with_labels=False)
plt.show()