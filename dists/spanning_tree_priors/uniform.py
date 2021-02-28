import numpy as np
from utils.Graph import Graph

# assumption: nodes labelled 0 to n-1
def randomTree(G):
    n = len(G)
    nodes = np.arange(n)
    visited = np.zeros(n, dtype=np.bool)
    hitTime = np.zeros(n, dtype=np.int)
    x = [np.random.choice(nodes)]
    visited[x[0]] = True
    hitTime[x[0]] = 0

    # random walk
    i = 0
    while(visited.sum() < n):
        nbh = G.GetNeighbours(x[i])
        r = np.random.choice(nbh)
        if (not visited[r]):
            hitTime[r] = i + 1
            visited[r] = True
            # end if
        x = x + [r]
        i = i+1
    # end random walk

    T = Graph(n)
    for i in range(n):
        if (i == x[0]): 
            continue
        p, q = hitTime[i]-1, hitTime[i]
        T.AddEdge(x[p], x[q])
    return T

class STPrior:
    def __init__(self, n, G=None):
        self._n = n
        if G:
            self._G = G
        else: 
            g = Graph(n) 
            g.SetComplete()
            self._G = g
    
    __name__ = 'uniform'
    
    def Sample(self):
        return randomTree(self._G)
    
    def PDF(self, param): 
        return 0
    
    def ParamType(self): 
        return self._Param.__name__