# PRADWIL

# Analysis of the Prader-Willi Syndrome Research Collaboration Network

## Abstract

The goal of this paper is to analyze the Prader-Willi syndrome (PWS) research community, through the lens of graph theory and social network analysis. In particular, the project focuses on the analysis of the co-authorship 
network within the scientific community dedicated to PWD research.

For this purpose, we collected a dataset containing all the published research on the PWS from the National Center for Biotechnology Information's PubMed database. 
We conducted a preliminary analysis of the dataset to study the evolution of the number of publications, the number of authors and the number of authors per paper over time.
Then, we constructed the aforementioned co-authorship network
and we characterized its structure by computing the main network metrics. 
We also investigated the scale-free property of the network by analyzing the power-law degree distribution.
Then, we identified the most influential authors in the largest connected component of the network by measuring centrality metrics, and we provided a final ranking using the Borda count method. 
Finally, we performed community detection on the largest connected component of the network using the Louvain algorithm.

## Introduction

Prader-Willi syndrome (PWS) is a genetic disorder and is recognized as the most common genetic cause of life-threatening childhood obesity.
According to the Foundation for Prader-Willi syndrome Research[^1], PWS occurs in approximately one out of every 15,000 births. It affects males and females with equal frequency and affects all races and ethnicities. 
Research findings serve as the foundation for advocacy efforts. It is crucial to raise awareness about PWS within the medical community, among policymakers, and in the general public. This is essential for garnering support and resources to advance research initiatives. Collaborative research on a global scale, involving researchers and healthcare professionals, can accelerate the pace of discovery, leading to more comprehensive insights and innovative solutions.

Graph theory, and particularly social network analysis, are crucial tools for evaluating the quality and effectiveness of research on Prader-Willi Syndrome.
In our project, we utilized tools developed in graph theory to systematically analyze the structure of the Prader Willi Syndrome research collaboration network.

The paper is organized as follows: 
in the Materials and Methods section, we provides details on data acquisition, computational tools and Python libraries employed in the analysis. Additionally, we describe the methods used for the characterization of the co-authorship network, the analysis of the scale free property of the network, the identification of the most influential nodes and the Louvain algorithm for community detection; in the Results section, we present and discuss the results of the analysis.
Finally, we reported our conclusions in the last section.

## Materials and Methods

### Data acquisition

We retrieved the data from the National Center for Biotechnology Information (NCBI), a division of the United States National Library of Medicine (NLM), specifically accessing the PubMed database[^2] . To collect the necessary articles for analysis, we utilized NCBI's e-utilities through a bash script on the UNIX command line [^3].

### Computational Tools and Python Frameworks Employed

All the code for data acquisition and plotting was executed on a machine with the following specifics:

| Hardware Component Specifics | Value |
| --- | --- |
| Processor | AMD Ryzen 5600X |
| Memory | 64GB DDR4 RAM (2400MHz, 4x16GB configuration) |
| Storage | 500GB NVMe SSD, Two 10TB HDDs |
| Graphics Processing Unit (GPU) | Nvidia RTX 2070 |
| Operating System | Linux Ubuntu 20.04 |

The computational framework utilized the capabilities of various software tools and libraries specifically chosen to meet the requirements of our research.
 
Notably, the the following Python libraries were employed for the network analysis and visualization:

- **NetworkX**: Utilized for the creation, manipulation, and analysis of complex networks and graph structures.
- **Matplotlib**: Employed for data visualization, including the creation of static, interactive, and animated plots.

### Construction of the authors collaboration network

We start by constructing the author-paper bipartite network $ G = (U, V, E)$, where the disjoint and independent sets of nodes $U$ and $V$ represent authors and papers, while the links between them denote the authorship relation. 
Subsequently, we derive the coauthorship collaboration network from the original bipartite network by projecting it onto the set of author nodes. 

In this new graph, denoted as $G' = (V, E)$, each author is represented by a vertex $v_i$, while the existence of an edge between two different authors means that there exists at least one path between them in the original bipartite graph $G$, indicating a shared paper.

We decided to employ a weighted projection of $G$ to obtain $G'$. The weight of each edge corresponds to the number of common nodes in the original bipartite graph $G$, reflecting the number of papers authors have published together. 

This network structure aligns with the concept that frequent collaborators should exhibit stronger connections in the coauthorship network compared to authors with fewer shared publications.

### Methods for the analysis of the authors collaboration network
#### Metrics for network characterization

The initial step in the analysis involves testing the connectivity of the graph and identifying its largest connected component.
Subsequently, we compute the following network metrics for the largest connected component:

- **Density**: 
The density of a graph is defined as the ratio between the number of edges in the graph and the maximum number of edges in a graph with the same number of nodes:

$$
D = \frac{2m}{n(n-1)}
$$

where $m$ is the number of edges in the graph and $n$ is the number of nodes in the graph.
The value of the density ranges from 0 to 1, and it is equal to 1 for a complete graph (a graph in which each node is connected to all other nodes), and it is equal to 0 for a graph without edges.

- **Average clustering coefficient (weighted and unweighted)**:

The local clustering coefficient in an undirected and unweighted graph for a node $i$ is defined as the fraction of potential triangles involving that node that actually exist in the graph, meaning the probability that two neighbors of the node $i$ are connected to each other. 
Mathematically, it is expressed as:

$$
C^{unw}_i = \frac{2t_i}{k_i(k_i-1)}
$$

where $t_i$ is the number of triangles through node $i$ and $k_i$ is the degree of node $i$.

On the other hand, there are several way for defining the local clustering coefficient in a weighted graph.
In our project, we employed the geometric average of the subgraph edge weights:

$$
C^{w}_u = \frac{1}{k_u(k_u-1)} \sum_{i,j} \sqrt[3]{\hat{w}_{ij} \hat{w}_{iu} \hat{w}_{ju}}
$$

where the edge weights $\hat{w}_{ij}$ are normalized by the maximum weight in the graph, and $k_u$ is the degree of the node $u$ and the value $C_u$ is set to 0 if $k_u < 2$.

In both the weighted and unweighted case, the global clustering coefficient is defined as the average of the local clustering coefficients of all the nodes in the graph:

$$
C = \frac{1}{n} \sum_{i=1}^n C_i
$$

The value of the clustering coefficient ranges from 0 to 1, and
an high value of the clustering coefficient indicates that many nodes in the graph tend to cluster together, while a low value indicates that nodes tend to be more isolated.

In our context, we measured the clustering coefficient of the largest connected component of the coauthorship 
collaboration network for both the weighted and unweighted case.

- **Average shortest path**

The average shortest path of the collaboration network is the average number of steps along the shortest paths for all possible pairs of network nodes. 
The mathematical expression for the average shortest path in the unweighted case is:

$$
L = \frac{1}{n(n-1)} \sum_{i \neq j} d(v_i, v_j)
$$

where $d(v_i, v_j)$ is the length of the shortest path between the nodes $v_i$ and $v_j$, and $n$ is the number of nodes in the network.
The average shortest path is a measure of the efficiency of information exchange in a network.

The previous definition can be extended to the weighted case as:

$$
L = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{w(v_i, v_j)}
$$

where $w(v_i, v_j)$ is the weight of the shortest path between the nodes $v_i$ and $v_j$.

According to the conventional definition, edge weights are typically interpreted as distances or costs, implying that shorter paths have lower weights. However, in our context, a higher weight between two nodes indicates a stronger collaboration between the two authors. To compute the aforementioned metrics, we need to establish a new weight scheme where the weights are defined as the reciprocals of the original weights.

#### The scale free property

One of the notable models for complex networks is the **scale-free network**, characterized by a degree distribution that follows a heavy-tailed power law. 
This implies an abundance of nodes with degrees significantly higher than the average, and this property is associated with the network's **robustness**. 
To investigate this, we analyzed the power-law degree distribution of the coauthorship collaboration network using methods outlined by Clauset et al., (2009).[^4]

The analysis involves the following steps:

1. Firstly, we fit the tail of the empirical distribution of the degree with a power-law distribution:

$$
p(d) \propto d^{-\alpha}
$$

Here, $\alpha$ is a constant parameter, typically $2 < \alpha < 3$. 
In our context, $ d $ represents the degrees of nodes, and $p(d)$ represents the probability degree distribution of the network, normalized to one. 
In most cases, the power law model is applicable only on the tail of the empirical distribution, 
meaning for degrees greater than a minimum $d_{min}$. 
The fitting function will be characherized by an estimated scaling parameter $\hat{\alpha}$ and the lower 
bound $d_{min}$ .
Then, we compute the value $D$ of the Kolmogorov-Smirnov (KS) statistics for this fit, which is interpreted as a "distance" between the empirical distribution and the fitted power law.

Then, in order to assess the goodness of the fit, we use the following procedure:

2. We generate a substantial number of synthetic datasets mimic the distribution of the empirical data below $d_{min}$  while following the fitted power law above $d_{min}$. 
In particular, we generate from the fitted power law a number of synthetic datasets equal to the number of elements in the original dataset which have degree greater than $d_{min}; while for the remaining elements we sample uniformly at random from the observed data set that have degree less than $d_{min}$.

3. We individually fit each synthetic dataset to its own power-law model and calculate the KS statistic for each 
one relative to its own model.

4. Finally, the goodness of the fit is assessed through the *p-value*,  which is computed as the fraction 
of times the KS statistics of the syntetic datases is larger than the observed KS distance. 
The *p-value* is therefore interpreted as a measure of the plausibility of the hypothesis that our data conforms 
to a power-law distribution. 

A large *p-value* suggests that the difference between empirical data and the model can be attributed to 
statistical fluctuations. Conversely, if the *p-value* is smaller than a specified threshold (in our case, $0.1$),
the model does not provide a plausible fit for the data, and the hypothesis is rejected.
To achieve accuracy to about two decimal places, we generate $2500$ synthetic sets. 

We performed the degree distribution analysis on the coauthorship collaboration network using the powerlaw package for Python.

#### Identification of the most influential nodes

The identification of the most influential nodes in a network is a fundamental task in network analysis, 
since it allows to identify the nodes that are most important for the structure and the functioning of the network.
There are many metrics that can be used to evaluate the importance of a node in a network, each of them
capturing a different aspect of the node's importance.
In this section, we will describe some of the most common metrics for node importance evaluation,
and we will use them to identify the most influential authors in the Alzheimer's disease collaboration network.

- Degree centrality

The degree centrality quantifies the importance of a node in a network by computing the degree of each node (
i.e. the number of links that the node has with other nodes in the network),
and then normalizing it by the maximum possible degree in the network (which is given by the number of nodes
minus one): 

$$
C_D(v) = \frac{k_v}{n-1}
$$

where $k_v$ is the degree of the node $v$ and $n$ is the number of nodes in the network.
The degree centrality assigns a higher score to the nodes with a higher degree, meaning that the nodes with
more links are considered more important.

- Betweenness Centrality

Betweenness centrality is a measure that assesses the importance of a node in a network by calculating the 
number of shortest paths passing through that node for all pairs of nodes in the network. 
The measure is then normalized by the maximum possible number of shortest paths between all pairs of nodes 
in the network. 
Mathematically, it is expressed as:

$$
C_B(v) = \frac{\sum_{s \neq v \neq t} \sigma_{st}(v)}{\sum_{s \neq t} \sigma_{st}}
$$

Here, $\sigma_{st}$ represents the number of shortest paths between nodes $s$ and $t$, and $\sigma_{st}(v)$ 
denotes the number of those paths that traverse the node $v$. Betweenness centrality identifies nodes that act 
as crucial bridges between different sections of the network, playing a pivotal role in the flow of information.

In our specific context, we need to consider the coauthorship collaboration network as a weighted graph. 
Consequently, when calculating shortest paths, we must treat paths with higher weights as "shortest", reflecting 
more frequent collaborations between authors. However, in algorithms for computing shortest paths, weights are 
often interpreted as distances or costs, implying that shorter paths have lower weights.

Therefore, in our calculations for betweenness centrality, we must account for the weighted nature of the graph 
by taking the reciprocal of the weights. 
This adjustment ensures that the algorithms correctly identify paths with the highest collaborative significance, 
aligning with the notion that heavier weights represent stronger connections between authors.

- Closeness centrality

The closeness centrality is defined as the inverse of the average distance between a node and all other nodes.
For each node $v$ in the network, the closeness centrality is computed by calculating the average of the distances 
from the node $v$ to all other nodes in the network (length of the shortest path between $v$ and the other nodes),
and then taking the reciprocal of this value:

$$
C_C(v) = \frac{1}{\frac{1}{n-1} \sum_{u \neq v} d(v,u)}
$$

where $d(v,u)$ is the length of the shortest path between the nodes $v$ and $u$, and $n$ is the number of nodes
in the network.

Closeness centrality provides a metric for evaluating how proximate a node is to all other nodes within a network. 
Nodes exhibiting high closeness centrality can efficiently reach all other nodes in the network in a limited number 
of steps. This measure is indicative of how rapidly information can disseminate from a particular node to the 
entire network.

Similar to the considerations for betweenness centrality, the computation of shorter paths in a weighted graph 
necessitates the adjustment of weights. 
Also in this case, we take the reciprocal of the weights to properly account for the weighted nature of the graph. 

- Eigenvector centrality

The eigenvector centrality measures the importance of a node in a network by considering the importance of 
its neighbors, providing a recursive definition of node importance.

The eigenvector centrality \(x_i\) for node \(i\) is defined as:

$$
x_i = \frac{1}{\lambda} \sum_k a_{k,i} \, x_k
$$

where \(A = (a_{i,j})\) represents the adjacency matrix of the network, \(\lambda \neq 0\) is a constant, 
and \(x_k\) is the centrality of node \(k\). The same relationship can be expressed in matrix form as:

$$
\lambda x = x A
$$

where \(\lambda\) is the eigenvalue and \(x\) is the eigenvector of the adjacency matrix \(A\).

Consequently, the eigenvector centrality is given by the eigenvectors associated with the largest eigenvalue 
of the adjacency matrix of the network.

For computing the eigenvector centrality of the coauthorship collaboration network, we employed the power 
iteration method. 
This iterative technique starts with a random vector and repeatedly multiplies it by the adjacency matrix 
of the network until the vector converges to the eigenvector associated with the largest eigenvalue of the 
adjacency matrix. At each iteration, the vector is normalized to prevent it from growing indefinitely.

- Final ranking with Borda count

After the evalution of the importance of each node in the network using the four metrics described above,
we combined the results of the four metrics to obtain a final ranking of the most influential authors in the
Alzheimer's disease collaboration network.

In order to combine the results, we used the Borda count method, which is a single-winner election method in
which voters rank candidates in order of preference.
In particular, for each metric, we ranked the authors in descending order according to the value of the metric,
and we assigned to each author a score equal to the number of authors that are ranked below him.
Then, we summed the scores obtained by each author for each metric, and we ranked the authors according to
the total score.


#### Community detection and Louvain algorithm

Community detection is the process of identifying groups of nodes that are more densely connected to each 
other than to the rest of the network. This can be useful in order to understand the structure of the network and to identify nodes wich shares similar
characteristics or functions.
In our context, communities represent groups of authors that have a higher tendency to collaborate with each other.
There is no universally accepted definition of what constituets a community, but there are several measures
that can be used to evaluate the quality of a community partition of a network.
In general, a good community partition is characterized by a high density of edges within communities and a 
low density of edges between communities.
So that, a measure of the quality of a community partition of a network is the modularity, which, for an undirected 
network, is defined as:

$$
Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$ 

where $A_{ij}$ is the element of the adjacency matrix of the network, $k_i$ and $k_j$ are the degrees of the 
nodes $i$ and $j$, $m$ is the number of edges in the network, $c_i$ and $c_j$ are the communities to which 
the nodes $i$ and $j$ belong, and $\delta(c_i, c_j)$ is the Kronecker delta function, which is equal to 1 
if $c_i = c_j$ (the nodes $i$ and $j$ belong to the same community) and 0 otherwise.
Modularity ranges from -1 to 1, and a value greater than 0.3 is generally considered as a good partition.
The modularity is positive if the number of edges within communities is greater than the expected number of
edges in a random network with the same degree distribution.
There are several algorithms for community detection, and many of them are based on the maximization of the
modularity.
In our project, we performed the community detection using the Louvain algorithm, which is a modularity-based,
agglomerative, heuristic method.
This algorithm, proposed by Blondel et al. in 2008, have been shown to be very fast and to produce partitions 
with a high modularity.
It consists of two phases: 
1. The algorithm starts by assigning each node to its own community. Then, for each node in the network, it
evaluates the gain in modularity that would result from moving the node to each of its neighbors' communities as:

$$
\Delta Q = \frac{1}{2m} \left[ \frac{\sum_{in} + k_{i,in}}{2m} - \left( \frac{\sum_{tot} + k_i}{2m} \right)^2 \right] - \frac{1}{2m} \left[ \frac{\sum_{in}}{2m} - \left( \frac{\sum_{tot}}{2m} \right)^2 - \left( \frac{k_i}{2m} \right)^2 \right]
$$

where $\sum_{in}$ is the sum of the weights of the links between the node $i$ and the nodes in the community
to which $i$ belongs, $\sum_{tot}$ is the sum of the weights of the links between the node $i$ and all the
nodes in the network, $k_i$ is the degree of the node $i$, $k_{i,in}$ is the sum of the weights of the links
between the node $i$ and the nodes in the community to which $i$ belongs, and $m$ is the sum of the weights
of all the links in the network.
The order im which the nodes does not have significant influence on the final modularity value, but it 
may affect the computational time. 
The node is then moved to the neighbor's community that results in the largest increase in modularity.
This process is repeated iteratively until no further increase in modularity can be achieved.

2. In the second phase, the algorithm builds a new network whose nodes are the communities found in the first
phase. The weights of the links between the communities are equal to the sum of the weights of the links
between the nodes in the two communities. The algorithm then repeats the first phase on this new network.

The two phases are repeated iteratively until a maximum of modularity is reached.

The Louvain algorithm is an agglomerative and hierarchical method, meaning that it starts from the nodes
and builds the communities from the bottom up.

## Results 

### Description of the dataset and preliminary analysis

We acquired from the PubMed database 4616 papers' ids related to Prader Willi Syndrome; for each of them, we extracted the authors' names and the year of publication.
The resulting dataset contains papers published from 1963 to the present day.
We performed a preliminary analysis of the dataset in order study the evolution of the number of publications,
the number of authors and the number of authors per paper over time.
The following figure show how these quantities evolved by a four-year window over the period 1963 - 2023.

![Pre_analysis](./images/pre_analysis_plots.png)

We observed that all the three quantities increased over time; in particular, the number of publications
followed a linear trend ($R^2 = 0.95$), while the number of authors followed an exponential trend.

### The coauthorship collaboration network

We constructed the coauthorship collaboration network from the bipartite network of authors and ids.
The resulting network contains ... edges and ... nodes, where ... are authors nodes and ... are paper nodes.
The following figure shows the resulting bipartite network:

![Bipartite network](./images/bipartite_network.png)

We then projected the bipartite network onto the set of author nodes, obtaining the weighted coauthorship collaboration network. 
The network contains ... edges and ... nodes and it is not connected.

(We found that the graph is not connected, and it is composed of 1442 connected components. 
We filtered out the components with one single node, and we found that the largest connected component contains
9289 nodes, which is about 56% of the total number of nodes in graph.)
We found that the collaboration network is composed of ... connected components, and the distribution of the
number of nodes in the connected components is shown in the following figure:

![Connected components](./images/connected_components.png)

The largest connected component contains ... nodes, which is about ...% of the total number of nodes in the network;
all the other connected components are much smaller, and they contain less than ... nodes.
We therefore decided to focus our analysis on the largest connected component.

### Metrics for network characterization

In order to characterize the structure of the coauthorship collaboration network, we computed the density, the
average clustering coefficient and the average shortest path of the largest connected component.
The clustering coefficient and the average shortest path were computed both for the weighted and unweighted case.
The results are summarized in the following table:

| Metric | Weighted | Unweighted |
| --- | --- | --- |
| Density | - | 0.002 |
| Average clustering coefficient | 0.023 | 0.882 |
| Average shortest path | 4.359 | 5.926 |

The low value of the density suggests that the coauthorship collaboration network is a sparse graph. The unweighted average clustering coefficient is quite high, indicating that the nodes in the network tend to cluster together. 

### The scale free property

Firstly, we estimate the scaling parameter $\hat{\alpha}$ and the lower bound $d_{\text{min}}$ of the fitted power law and we compute the value $D$ of the Kolmogorov-Smirnov (KS) statistics for this fit.
The results are summarized in the following table:

| Parameter | Value |
| --- | --- |
| $\hat{\alpha}$ |  |
| $d_{\text{min}}$ |  |
| $D$ |  |

The plot of the empirical degree distribution and the cumulative degree distribution are shown in the following figures:

![Degree distribution](./images/degree_distribution.png)

![CDF](./images/cumulative_degree_distribution.png)

After that, we estimate the *p-value* of the goodness of the fit, which is found to be equal to $p = $.
The large *p-value* suggests that the difference between empirical data and the model can be attributed to
statistical fluctuations, and therefore the model provides a plausible fit for the data. 

### Identification of the most influential nodes

We identified the most influential authors in the coauthorship collaboration network by compunting the degree
centrality, the betweenness centrality, the closeness centrality and the eigenvector centrality of each node.
These metrics are used to evaluate the importance of a node in a network by capturing different aspects of
the node's importance, the final ranking is obtained by combining the results of the four metrics using the
Borda count method.
In the following table, we show the top 10 authors according to each metric and the final ranking:

| Degree centrality | Betweenness centrality | Closeness centrality | Eigenvector centrality | Borda score |
| --- | --- | --- | --- | --- |
| Butler Merlin G | Tauber Maithé | Poitou Christine | Grugni Graziano | Poitou Christine |
| Grugni Graziano | Muscatelli F | Coupaye Muriel | Crinò Antonino | Grugni Graziano |
| Crinò Antonino | Molinas Catherine | Tauber Maithé | Sartorio Alessandro | Crinò Antonino |
| Miller Jennifer L | Lalande M | Goldstone Anthony P | Butler Merlin G | Butler Merlin G |
| Tauber Maithé | Poitou Christine | Grugni Graziano | Poitou Christine | Miller Jennifer L |
| Poitou Christine | Goldstone Anthony P | Crinò Antonino | Pellikaan Karlijn | Goldstone Anthony P |
| Driscoll Daniel J | Butler Merlin G | Pellikaan Karlijn | Coupaye Muriel | Coupaye Muriel |
| Horsthemke B | Miller Jennifer L | Molinas Catherine | Miller Jennifer L | Driscoll Daniel J |
| Nicholls R D | Nicholls R D | Caixàs Assumpta | Caixàs Assumpta | Tauber Maithé |
| Haqq Andrea M | Horsthemke B | Sartorio Alessandro | Rosenberg Anna G W | Caixàs Assumpta |

### Community detection and Louvain algorithm

We performed the community detection using the Louvain algorithm (for reproducibility, we fixed the random seed to 42). The algorithm identified 73 different communities, and the distribution of the number of nodes in the communities is shown in the following figure:

![Community size](./images/community_size.png)

The largest community contains ... nodes, which is about ...% of the total number of nodes in the network.
The modularity of the final partition is about $Q = 0.904$.


## Discussion

## Conclusion

## References

[^1] : Foundation for Prader-Willi Syndrome Research. What is Prader-Willi Syndrome? [Link](https://www.fpwr.org/what-is-prader-willi-syndrome#definition) (Accessed: December 5, 2023)

[^2] : National Center for Biotechnology Information. PubMed. [Link](https://pubmed.ncbi.nlm.nih.gov/) (Accessed: December 5, 2023)

[^3] : Entrez Programming Utilities. [Link](https://www.ncbi.nlm.nih.gov/books/NBK179288/) 

[^4] : Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. [DOI](
https://doi.org/10.48550/arXiv.0706.1062)

