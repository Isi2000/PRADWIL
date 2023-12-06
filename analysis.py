import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import numpy as np
import analysis_functions as af

# Read the data-------------------------------------------------------------

df = pd.read_json("./data/data.json")

#Building the bipartite graph------------------------------------------------

print('Creating the author-paper bipartite graph...')

G = af.create_bipartite_graph(df)

print('Done!')

number_of_authors_nodes = len([node for node in G.nodes if G.nodes[node]['bipartite'] == 1])
number_of_papers_nodes = len([node for node in G.nodes if G.nodes[node]['bipartite'] == 0])
total_number_of_nodes = number_of_authors_nodes + number_of_papers_nodes
number_of_edges = G.number_of_edges()

print('Number of authors nodes:', number_of_authors_nodes)
print('Number of papers nodes:', number_of_papers_nodes)
print('Total number of nodes:', total_number_of_nodes)
print('Total number of edges:', number_of_edges)

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
print('Density: ', nx.density(C))
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
      len(max(connected_components, key=len)) / number_of_nodes)

# Computing the length of all the connected components

lengths = [len(comp) for comp in filtered_connected_components]
lenghts_not_filtered = [len(comp) for comp in connected_components]
sorted_lengths_not_filtered = sorted(lenghts_not_filtered, reverse=True)
sorted_lengths = sorted(lengths, reverse=True)

print('Lengths of the connected components: ', sorted_lengths)

# Histogram of the lengths of the connected components with a logarithmic scale on the y axis

import matplotlib.pyplot as plt

#Escludo il primo elemento perché è il numero di nodi della componente connessa più grande
filtered_sorted_lengths = sorted_lengths_not_filtered[1:]

plt.figure(figsize=(10, 6))
plt.hist(filtered_sorted_lengths, bins=115, log=True)
plt.xlim(0, 115)
plt.yscale('log')
plt.xlabel('Number of nodes', fontweight='bold')
plt.xticks(np.arange(0, 116, 5))
plt.ylabel('Number of connected components', fontweight='bold')
plt.title('Histogram of the number of nodes of the connected components', fontsize=15, fontweight='bold')

#Saving the histogram
plt.savefig('./images/connected_components_histogram.png')

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
avg_shortest_path = nx.average_shortest_path_length(cc, weight= lambda u, v, d: 1 / d['weight'] )
print('Average shortest path:', avg_shortest_path)
avg_shortest_path_unweighted = nx.average_shortest_path_length(cc)
print('Average shortest path (unweighted):', avg_shortest_path_unweighted)


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





