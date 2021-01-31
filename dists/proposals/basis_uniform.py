import random 
from itertools import combinations

class Proposal:
    def __init__(self, n, Param):
        self._n = n          
        self._Param = Param
        
    def Sample(self, param):
        i = random.choice(range(len(param._basis)))
        param_ = param.copy()
        param_.BinAddOneBasis(i)
        return param_ 
    
    def PDF(self, p0, p1):
        return 0 
    def ParamType(self): 
        return self._Param.__name__