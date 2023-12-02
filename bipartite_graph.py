import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import analysis_functions as af
import random

# Leggi i dati
df = pd.read_json("data.json")

# Crea il grafo bipartito
G = af.create_bipartite_graph(df)

# Estrai i nodi autori e i nodi papers
authors_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 1}
papers_nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}

authors_nodes_subset = set(random.sample(authors_nodes, 20))
# Nodi paper collegati ai nodi autori
authors_nodes_subset_papers = set()
for node in authors_nodes_subset:
    authors_nodes_subset_papers.update(list(G.neighbors(node)))

# Sottografo dei nodi autori e dei nodi paper collegati
authors_nodes_subset_subgraph = G.subgraph(authors_nodes_subset.union(authors_nodes_subset_papers)).copy()

print('Number of nodes in the subgraph: ', len(authors_nodes_subset_subgraph.nodes))

# Calcola i gradi dei nodi autori
node_degrees = dict(G.degree(authors_nodes))

# Trova i 10 nodi autori con il grado più alto
#top_10_nodes = sorted(authors_nodes, key=lambda node: node_degrees[node], reverse=True)[:10]

# Imposta le caratteristiche del grafico
pos = nx.bipartite_layout(G, nodes=authors_nodes)
node_color = 'skyblue'
node_size = 0.1
edge_width = 0.005

# Crea la figura principale
fig, main_ax = plt.subplots()

# Disegna il grafo principale
nx.draw(G, pos, nodelist=authors_nodes, with_labels=False,
        node_color=node_color, node_size=node_size, width=edge_width,
        ax=main_ax)
main_ax.set_title('Grafo Principale')

pos1 = nx.bipartite_layout(authors_nodes_subset_subgraph, nodes=authors_nodes_subset)

# Crea l'area ingrandita con i primi 10 autori
inset_ax = fig.add_axes([0.65, 0.15, 0.3, 0.3], facecolor='white')

# Impostazione dei colori
node_color = ['red' if G.nodes[n]['bipartite'] == 0 else 'blue' for n in G.nodes]
node_color_subset = ['red' if G.nodes[n]['bipartite'] == 0 else 'blue' for n in authors_nodes_subset_subgraph.nodes]

# Disegna i nodi autori in rosso
nx.draw_networkx(authors_nodes_subset_subgraph, pos1, nodelist=authors_nodes_subset,
                node_color='skyblue', with_labels=False,
                node_size=node_size, width = 0.1, ax=inset_ax)

# Disegna gli archi
nx.draw_networkx_edges(authors_nodes_subset_subgraph, pos1, width=edge_width, ax=inset_ax)

inset_ax.set_title('Zoom')

# Riduci lo spazio vuoto attorno alla figura principale
#plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Mostra il grafico
plt.show()

