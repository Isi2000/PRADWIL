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

# Read the data-------------------------------------------------------------

df = pd.read_json("data.json")

# Pre analysis of the data-------------------------------------------------

df['Year'] = df['Date'].apply(af.convert_year)

df = af.add_year_interval(df, interval_length = 5)

# Computing the number of articles and authors per year interval
print('Computing the number of articles and authors per year interval...')

num_articles_per_interval = df['YearInterval'].value_counts().sort_index()

df['NumAuthors'] = df['Authors'].apply(af.count_authors)
num_authors_per_interval = df.groupby('YearInterval')['NumAuthors'].sum().sort_index()

result_df = pd.DataFrame({
    'YearInterval': num_articles_per_interval.index,
    'NumArticles': num_articles_per_interval.values,
    'NumAuthors': num_authors_per_interval.values
})

# Computing the average number of authors per article
result_df['AvgAuthorsPerArticle'] = result_df['NumAuthors'] / result_df['NumArticles']

# Saving the results in a JSON file

os.makedirs('./results', exist_ok=True)

print('Saving the results...')
result_df.to_json('./results/num_paper_authors.json', orient='records', lines=True, index = True)
print('Results saved!')

#Building the bipartite graph------------------------------------------------

print('Creating the author-paper bipartite graph...')

G = af.create_bipartite_graph(df)

print('Done!')

number_of_authors_nodes = len([node for node in G.nodes if G.nodes[node]['bipartite'] == 1])
# Non ho proprio idea del perchè mi dia un numero di paper diverso da quello che mi aspetto:
# più basso.
# Tocca controllare sta cosa quando si avranno i dati veri. 
# - > Mi è venuto in mente potrebbero essere le parti non connesse del grafo.
number_of_papers_nodes = len([node for node in G.nodes if G.nodes[node]['bipartite'] == 0])
total_number_of_nodes = number_of_authors_nodes + number_of_papers_nodes

print('Number of authors nodes:', number_of_authors_nodes)
print('Number of papers nodes:', number_of_papers_nodes)
print('Total number of nodes:', total_number_of_nodes)

# Visualizing the bipartite graph--------------------------------------------
# Secondo me ci conviene riportare in relazione solo una percentuale del vero 
# grafo, perchè altrimenti non si vede niente.

# bipartite layout
authors_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
authors_nodes_subset = set(random.sample(authors_nodes, 50))
authors_nodes_subset_subgraph = G.subgraph(authors_nodes_subset).copy()

pos = nx.bipartite_layout(G, nodes= authors_nodes_subset)

node_size = [deg for _, deg in authors_nodes_subset_subgraph.degree()]
edge_width = 0.1

nx.draw(G, pos, nodelist=authors_nodes_subset, with_labels=False,
        node_color='skyblue', node_size=node_size, width=edge_width)
plt.show()

# Poi dobbiamo salvare l'immagine

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

# Smallworldness property----------------------------------------------------
# Questa parte la lascio commentata perchè ci mette troppo tempo.
# In più non è nemmeno completa.

#sigma = nx.sigma(cc, niter=10, nrand=10)
#print('Smallworldness property:', sigma)

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

#Plotting the collaboration network-------------------------------------------

"""
with tqdm(total=100, desc="Plotting", position=0, leave=True) as pbar:
    def update_progress(*args, **kwargs):
        pbar.update(1)

    # Aggiorna la funzione di progresso di Matplotlib
    plt.show = update_progress

    # Plot del grafo di collaborazione
    pos = nx.spring_layout(cc)
    nx.draw(cc, pos, node_size=0.1, with_labels=False)

    # Chiudi la barra di avanzamento alla fine
    pbar.close()

plt.show()
"""

# Adding attributes to the nodes---------------------------------------------

# Adding the field of research attribute just for testing purposes
for node in G.nodes:
    field_of_research_value = random.randint(1, 100)
    G.nodes[node]['campo_di_ricerca'] = field_of_research_value

# Characterizing the network with Social Network metrics

field_of_research_assortativity = nx.attribute_assortativity_coefficient(G, 'campo_di_ricerca')
print('Field of research assortativity:', field_of_research_assortativity)

mixing_matrix = nx.attribute_mixing_matrix(G, 'campo_di_ricerca')
print('Mixing matrix:', mixing_matrix)
plt.pcolor( mixing_matrix, cmap = 'hsv' )
plt.show()

#----------------------------------------------------------------------------
# Identification of the most influential nodes
"""

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
"""
# Louvain communities---------------------------------------------------------

partition = community.best_partition(cc, weight='weight')

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

# Testing if there is a correlation between communities and field of research---
# Build the contingency table

print('Computing the contingency table...')




