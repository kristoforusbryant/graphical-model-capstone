import numpy as np
from utils.Graph import Graph

def get_all_graphs(n):
    G_list = []
    m = int(n * (n-1) / 2)
    triu_idx = np.triu_indices(n,1)
    for i in range(np.power(2, m)):
        b = format(i,'0' + str(m) + 'b')
        G_list.append(Graph(n))
        for j in range(len(b)):
            if int(b[j]):
                G_list[-1].AddEdge(triu_idx[0][j], triu_idx[1][j])
    return G_list