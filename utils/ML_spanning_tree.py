from .Graph import Graph
import numpy as np
import networkx as nx

# Based on (Edwards 2010) https://doi.org/10.1186/1471-2105-11-18

def BIC(i, j, n_obs, corr):
    mutual_information = - n_obs * np.log(1 - corr[i,j] ** 2) / 2
    return mutual_information - np.log(n_obs) / 2

def score_edges(data):
    n_obs, n = data.shape
    if n_obs == 0:
        return [ 1 for i, j in zip(*np.triu_indices(n, k=1)) ] #TODO: fix default for 0 and 1
    corr = np.corrcoef(np.transpose(data))
    return [ BIC(i, j, n_obs, corr) for i, j in zip(*np.triu_indices(n, k=1)) ]

def ML_spanning_tree(data):
    n = data.shape[1]
    triu = list(zip(*np.triu_indices(n, k=1)))
    scores = score_edges(data)

    G = nx.empty_graph(n)
    for i in range(n * (n - 1) // 2):
        G.add_edge( triu[i][0], triu[i][1], weight=scores[i])
    T = nx.maximum_spanning_tree(G, algorithm="kruskal")

    return Graph(n, dol=nx.to_dict_of_lists(T)), scores

def ML_forest(data):
    n = data.shape[1]
    triu = list(zip(*np.triu_indices(n, k=1)))
    scores = score_edges(data)

    G = nx.empty_graph(n)
    for i in range(n * (n - 1) // 2):
        G.add_edge( triu[i][0], triu[i][1], weight=scores[i])

    T = nx.maximum_spanning_tree(G, algorithm="kruskal")

    # Pruning
    S = np.zeros((n, n))
    S[np.triu_indices(n, k=1)] = scores
    S += np.transpose(S)

    edge_l = list(T.edges)
    for i, j in edge_l:
        if S[i, j] < 0. :
            T.remove_edge(i, j)

    return Graph(n, dol=nx.to_dict_of_lists(T))














