import numpy as np
# utils should be inside the path of the file that imports this

class Prior:
    def __init__(self, n, Param, basis):
        self._n = n
        self._Param = Param 
        self._basis = basis 
        
    def Sample(self):
        param = self._Param(self._n, basis=self._basis)
        for i in range(len(param._basis)): 
            if np.random.uniform() > 0.5:
                param.BinAddOneBasis(i)
        return param
    
    def PDF(self, param): 
        return 0
    def ParamType(self): 
        return self._Param.__name__