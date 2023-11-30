import networkx as nx
import community
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

# Read the data-------------------------------------------------------------

# These data are just for testing purposes
num_articoli = 5000
articoli_ids = range(1, num_articoli + 1)

autori_per_articolo = [random.sample(range(1, 10000), random.randint(1, 5)) for _ in range(num_articoli)]

date_pubblicazione = [datetime(2020, 1, 1) + timedelta(days=random.randint(1, 10000)) for _ in range(num_articoli)]

data = {'Id': articoli_ids, 'Authors': autori_per_articolo, 'Dates': date_pubblicazione}
rows = len(data['Id'])
print('Number of rows:', rows)
df = pd.DataFrame(data)

# Pre analysis of the data-------------------------------------------------

def convert_year(year):
    """
    Convert a string representing a year into an integer.
    
    Parameters
    ----------
    year : str
        The year to convert.
    
    Returns
    -------
    int
        The year converted into an integer.
    
    Raises
    ------
    ValueError
        If the year cannot be converted into an integer.
        
    """
    try:
        return int(year)
    except ValueError:
        return pd.NaT

def count_authors(authors_list):
    """
    Count the number of authors of a paper.

    Parameters
    ----------
    authors_list : list
        The list of authors of a paper.
    
    Returns
    -------
    int
        The number of authors of the paper.
    
    """
    return len(authors_list)

# Adding the year interval column
df['Dates'] = df['Dates'].astype(str)
df['Year'] = df['Dates'].str.split('-', expand=True)[0]

df['Year'] = df['Year'].apply(convert_year)

# Adding the year interval column
interval_length = 3
df['YearInterval'] = (df['Year'] // interval_length) * interval_length

# Computing the number of articles and authors per year interval
print('Computing the number of articles and authors per year interval...')
num_articles_per_interval = df['YearInterval'].value_counts().sort_index()

df['NumAuthors'] = df['Authors'].apply(count_authors)
num_authors_per_interval = df.groupby('YearInterval')['NumAuthors'].sum().sort_index()

result_df = pd.DataFrame({
    'YearInterval': num_articles_per_interval.index,
    'NumArticles': num_articles_per_interval.values,
    'NumAuthors': num_authors_per_interval.values
})

# Computing the average number of authors per article
result_df['AvgAuthorsPerArticle'] = result_df['NumAuthors'] / result_df['NumArticles']

# Saving the results in a JSON file

"""
print('Saving the results...')
result_df.to_json('./results/num_paper_authors.json', orient='records', lines=True, index = True)
print('Results saved!')
"""

#Building the bipartite graph------------------------------------------------

print('Creating the author-paper bipartite graph...')

G = nx.Graph()

num_paper = 0
# Add nodes and edges from the DataFrame
for _, row in df.iterrows():
    node_id = row['Id']
    authors = row['Authors']
    G.add_node(node_id, bipartite=0)  # bipartite=0 for 'Id' nodes
    num_paper += 1
    if len(authors) > 0:
        for author in authors:
            G.add_node(author, bipartite=1)  # bipartite=1 for 'Authors' nodes
            G.add_edge(node_id, author)

print('Done!')
print(num_paper)

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

subset_nodes = list(G.nodes)[:500]

# bipartite layout
pos = nx.bipartite_layout(G, subset_nodes)

node_size = 0.5
edge_width = 0.1

nx.draw(G, pos, nodelist=subset_nodes, with_labels=False,
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

"""
print('Saving the results...')
np.save('./results/degree_sequence.npy', degree_sequence)
print('Degree sequence saved!')
"""

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

print('Computing the Louvain communities...')
partition = community.best_partition(cc, weight='weight')

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




