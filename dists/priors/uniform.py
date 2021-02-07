import numpy as np
from itertools import combinations
# utils should be inside the path of the file that imports this

class Prior:
    def __init__(self, n, Param, basis=None):
        self._n = n
        self._Param = Param
        if basis is None:
            basis = []
            for idx in list(combinations(range(n), 2)): 
                g = Param(n)
                g.AddEdge(idx[0], idx[1])
                basis.append(g.copy())
        self._basis = basis
    __name__ = 'uniform'
    def Sample(self):
        param = self._Param(self._n)
        for i in range(self._n): 
            for j in range(i+1, self._n):
                if np.random.uniform() > 0.5:
                    param.AddEdge(i,j)
        return param
    def PDF(self, param): 
        return 0
    def ParamType(self): 
        return self._Param.__name__