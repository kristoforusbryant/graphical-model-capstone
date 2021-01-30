import numpy as np
# utils should be inside the path of the file that imports this

class Prior:
    def __init__(self, n, Param):
        self._n = n
        self._Param = Param 
        
    def Sample(self):
        param = self._Param(self._n)
        for i in range(self._n): 
            for j in range(i+1, self._n):
                if np.random.uniform() > 0.5:
                    param.AddEdge(i,j)
        return param
    def PDF(self, param): 
        return 0
    def ParamType(): 
        return self._Param.__name__