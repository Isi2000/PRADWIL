import networkx as nx
import community
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta
from tqdm import tqdm
df = pd.read_json("data.json")
print(df)

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
df['Date'] = df['Date'].astype(str)
df['Year'] = df['Date'].str.split('-', expand=True)[0]

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

def dijkstra_average_time(graph, num_iterations=1000):
    """
    Calculate Dijkstra's algorithm between two random nodes many times and estimate the average time.

    Parameters
    ----------
    graph : networkx.Graph
        The graph on which to perform Dijkstra's algorithm.
    num_iterations : int, optional
        The number of iterations to perform, by default 100.

    Returns
    -------
    float
        The average time taken for Dijkstra's algorithm.

    """
    total_time = 0

    for _ in tqdm(range(num_iterations)):
        # Select two random nodes
        nodes = random.sample(graph.nodes, 2)
        source_node, target_node = nodes[0], nodes[1]

        # Measure the time for Dijkstra's algorithm
        start_time = datetime.now()
        nx.shortest_path(graph, source=source_node, target=target_node)
        end_time = datetime.now()

        # Calculate and accumulate the time
        elapsed_time = (end_time - start_time).total_seconds()
        total_time += elapsed_time

    # Calculate average time
    average_time = total_time / num_iterations
    return average_time

# ... (your existing code)

# Example usage of the dijkstra_average_time function
average_dijkstra_time = dijkstra_average_time(cc)
print(f'Average time for Dijkstra\'s algorithm: {average_dijkstra_time} seconds')
