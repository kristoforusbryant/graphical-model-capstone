import numpy as np
from utils.Graph import Graph
from utils.generate_basis import hub

def randomTree(G):
    # assumption: nodes labelled 0 to n-1
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

class Uniform:
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

    def ParamType(self):
        return self._Param.__name__

class Hubs:
    def __init__(self, n, G=None):
        self._n = n
        if G:
            self._G = G
        else:
            g = Graph(n)
            g.SetComplete()
            self._G = g

    __name__ = 'hubs'

    def Sample(self):
        v = np.random.choice(np.arange(self._n))
        return Graph(self._n, hub(self._n, v))

    def ParamType(self):
        return self._Param.__name__

class Paths:
    def __init__(self, n, G=None):
        self._n = n
        if G:
            self._G = G
        else:
            g = Graph(n)
            g.SetComplete()
            self._G = g

    __name__ = 'hamiltonian_path'

    def Sample(self):
        p = np.random.permutation(self._n)
        T = Graph(self._n)
        for i in range(self._n - 1):
            T.AddEdge(p[i], p[i+1])
        return T

    def ParamType(self):
        return self._Param.__name__

class Fixed:
    def __init__(self, n, T):
        self._n = n
        self._T = T

    __name__ = 'fixed'

    def Sample(self):
        return self._T

    def ParamType(self):
        return self._Param.__name__
