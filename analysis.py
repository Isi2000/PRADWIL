import networkx as nx
import community
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from tqdm import tqdm
import numpy as np
import analysis_functions as af

from collections import defaultdict

# Read the data-------------------------------------------------------------

df = pd.read_json("./data/data.json")

#Building the bipartite graph------------------------------------------------

print('Creating the author-paper bipartite graph...')

G = af.create_bipartite_graph(df)

print('Done!')

number_of_authors_nodes = len([node for node in G.nodes if G.nodes[node]['bipartite'] == 1])
number_of_papers_nodes = len([node for node in G.nodes if G.nodes[node]['bipartite'] == 0])
total_number_of_nodes = number_of_authors_nodes + number_of_papers_nodes

print('Number of authors nodes:', number_of_authors_nodes)
print('Number of papers nodes:', number_of_papers_nodes)
print('Total number of nodes:', total_number_of_nodes)

# Coauthorship graph---------------------------------------------------------

print('Projecting the graph on the authors nodes (collaboration network)... ')

article_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
authors_nodes = set(G) - article_nodes

C = bipartite.weighted_projected_graph(G, authors_nodes) #weighted projection

print('Done!')

# Analysing the coauthorship graph-------------------------------------------

number_of_nodes = C.number_of_nodes()
number_of_edges = C.number_of_edges()

print('Number of nodes (authors): ', number_of_nodes)
print('Number of edges: ', number_of_edges)
# Testing if the graph is connected
print('Is connected:', nx.is_connected(C))

# Analysis of the graph components-------------------------------------------

number_of_connected_components = nx.number_connected_components(C)
connected_components = list(nx.connected_components(C))

# Filtering out the single nodes
filtered_connected_components = \
    [comp for comp in connected_components if len(comp) > 1] 

print('Number of connected components: ', number_of_connected_components)
print('Number of connected components with more than one node: ', len(filtered_connected_components))
print('Percentage of nodes in the largest connected component: ', \
      len(max(filtered_connected_components, key=len)) / number_of_nodes)

# Computing the length of all the connected components

lengths = [len(comp) for comp in filtered_connected_components]
sorted_lengths = sorted(lengths, reverse=True)

print('Lengths of the connected components: ', sorted_lengths)

largest_cc = max(filtered_connected_components, key=len)
print('Number of nodes of the largest connected component:', len(largest_cc))

#Building a network of the largest connected component-----------------------

cc = C.subgraph(largest_cc).copy() #ATTENTO QUI STO FACENDO UNA COPIA

# Acquiring degree sequence--------------------------------------------------

print('Computing the degree sequence...')
degree_sequence = sorted((d for n, d in C.degree()), reverse=True)

print('Saving the results...')
np.save('./results/degree_sequence.npy', degree_sequence)
print('Degree sequence saved!')


#-----------------------------------------------------------------------------
# Characterizing the network

density = nx.density(cc)
print('Density: ', density)
cluster_coeff = nx.average_clustering(cc, weight = 'weight')
print('Clustering coefficient:', cluster_coeff)
cluster_coeff_unweighted = nx.average_clustering(cc)
print('Clustering coefficient (unweighted):', cluster_coeff_unweighted)
#avg_shortest_path = nx.average_shortest_path_length(cc, weight= lambda u, v, d: 1 / d['weight'] )
#print('Average shortest path:', avg_shortest_path)
#avg_shortest_path_unweighted = nx.average_shortest_path_length(cc)
#print('Average shortest path (unweighted):', avg_shortest_path_unweighted)


#----------------------------------------------------------------------------
# Identification of the most influential nodes

# 1. Degree centrality

print('Ranking of authors by degree centrality:-------------------------------------------------- ')
degree_centrality = nx.degree_centrality(cc)
degree_centrality_sorted = sorted(degree_centrality.items(), key = lambda x: x[1], reverse=True)
for influential_author in degree_centrality_sorted[:10]:
    print('Author: ', influential_author[0], 'Degree centrality: ', influential_author[1])

# 2. Betweenness centrality

print('Ranking of authors by betweenness centrality:--------------------------------------------- ')
betweenness_centrality = nx.betweenness_centrality(cc, weight= lambda u, v, d: 1 / d['weight'])
betweenness_centrality_sorted = sorted(betweenness_centrality.items(), key = lambda x: x[1], reverse=True)
for influential_author in betweenness_centrality_sorted[:10]:
    print('Author: ', influential_author[0], 'Betweenness centrality: ', influential_author[1])

# 3. Closeness centrality
print('Ranking of authors by closeness centrality:------------------------------------------------')
closeness_centrality = nx.closeness_centrality(cc, distance= lambda u, v, d: 1 / d['weight'])
closeness_centrality_sorted = sorted(closeness_centrality.items(), key = lambda x: x[1], reverse=True)
for influential_author in closeness_centrality_sorted[:10]:
    print('Author: ', influential_author[0], 'Closeness centrality: ', influential_author[1])

# 4. Eigenvector centrality
print('Ranking of authors by eigenvector centrality:-----------------------------------------------')
eigenvector_centrality = nx.eigenvector_centrality(cc, weight= 'weight')
eigenvector_centrality_sorted = sorted(eigenvector_centrality.items(), key = lambda x: x[1], reverse=True)
for influential_author in eigenvector_centrality_sorted[:10]:
    print('Author: ', influential_author[0], 'Eigenvector centrality: ', influential_author[1])

# Borda count

def Borda_score(descending_list_of_tuples):
    Borda_score = {}
    for i, author in enumerate(descending_list_of_tuples):
        Borda_score[author[0]] = len(descending_list_of_tuples) - 1 - i
    return Borda_score

Borda_score_degree = Borda_score(degree_centrality_sorted)
Borda_score_betweenness = Borda_score(betweenness_centrality_sorted)
Borda_score_closeness = Borda_score(closeness_centrality_sorted)
Borda_score_eigenvector = Borda_score(eigenvector_centrality_sorted)

Borda_score_list = [Borda_score_degree, Borda_score_betweenness, \
                    Borda_score_closeness, Borda_score_eigenvector]

Borda_score_sum = {}

for author in Borda_score_degree.keys():
    Borda_score_sum[author] = sum([Borda_score[author] for Borda_score in Borda_score_list])

Borda_score_sum_sorted = sorted(Borda_score_sum.items(), key = lambda x: x[1], reverse=True)

print('Ranking of authors by Borda score:----------------------------------------------------------')
for influential_author in Borda_score_sum_sorted[:10]:
    print('Author: ', influential_author[0], 'Borda score: ', influential_author[1])

# Louvain communities---------------------------------------------------------

partition = community.best_partition(cc, weight='weight')

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

print('Length of the communities: ', [len(community) for community in list_of_communities])
print('Percentage of nodes in the main community: ', len(main_community) / number_of_nodes)
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

# Generare il grafico con tutte le comunità colorate in modo diverso 

pos = nx.spring_layout(cc)

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

# Crea un dizionario di etichette per i primi 10 nodi
labels = {node: node for node in top_10_nodes}

# Imposta la grandezza dei label
label_font_size = 6

nx.draw(cc, pos, with_labels=False, labels = labels, font_size=label_font_size,
        node_size=node_size, width=edge_width, font_color='black', alpha=0.7,
        node_color=list(partition.values()))
plt.show()

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





