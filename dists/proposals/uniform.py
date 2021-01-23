import random 
from itertools import combinations
from dists.params.graph import Param 

class Proposal:
    def __init__(self,n):
        self._n = n          
    def Sample(self, param):
        i,j = random.choice(list(combinations(list(param.GetDOL().keys()), 2)))
        param_ = param.copy()
        param_.FlipEdge(i,j)
        return param_ 
    def PDF(self, p0, p1):
        return 0 
    def ParamType(): 
        return Param.__name__