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
    
    __name__ = 'hubs'
    
    def Sample(self):
        v = np.random.choice(np.arange(self._n))
        return Graph(self._n, hub(self._n, v))
    
    def PDF(self, param): 
        return 0
    
    def ParamType(self): 
        return self._Param.__name__