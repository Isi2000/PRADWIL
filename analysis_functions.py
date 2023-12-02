import pandas as pd
import networkx as nx


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

def add_year_interval(data, interval_length):
    """
    This function adds a column to the dataframe containing the interval of years to which the paper belongs.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe containing the papers.
        Note that the dataframe must contain a column named 'Year' containing the year of publication of the paper.
    interval_length : int
        The length of the interval of years.
        For example, if interval_length = 5, the interval of years will be 2010-2014, 2015-2019, etc.
    
    Returns
    -------
    pandas.DataFrame
        The dataframe containing the papers with the new column named 'YearInterval'.
    """

    data['YearInterval'] = (data['Year'] // interval_length) * interval_length
    return data

def create_bipartite_graph(df):
    """
    This function creates the bipartite graph of authors and papers.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the papers.
        Note that the dataframe must contain a column named 'Id' containing the id of the paper and a column named 'Authors' containing the list of authors of the paper.

    Returns
    -------
    networkx.Graph
        The bipartite graph of authors and papers.
    """
    G = nx.Graph()

    # Add nodes and edges from the DataFrame
    for _, row in df.iterrows():
        node_id = row['Id']
        authors = row['Authors']
        G.add_node(node_id, bipartite=0)  # bipartite=0 for 'Id' nodes
        if len(authors) > 0:
            for author in authors:
                G.add_node(author, bipartite=1)  # bipartite=1 for 'Authors' nodes
                G.add_edge(node_id, author)

    return G