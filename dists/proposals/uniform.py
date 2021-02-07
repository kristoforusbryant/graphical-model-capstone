import random 
from itertools import combinations

class Proposal:
    def __init__(self, n, Param):
        self._n = n          
        self._Param = Param
    __name__ = 'uniform'
       
    def Sample(self, param):
        i,j = random.choice(list(combinations(list(param.GetDOL().keys()), 2)))
        param_ = param.copy()
        param_.FlipEdge(i,j)
        return param_ 
    def PDF(self, p0, p1):
        return 0 
    def ParamType(self): 
        return self._Param.__name__