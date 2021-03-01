import numpy as np
from utils.Graph import Graph
from utils.generate_basis import hub

class STPrior:
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
    
    def PDF(self, param): 
        return 0
    
    def ParamType(self): 
        return self._Param.__name__