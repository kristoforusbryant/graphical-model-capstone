from utils.Graph import Graph
from itertools import combinations
import numpy as np
import copy
from utils.generate_basis import cut_basis

# (Graph, Basis, Active Basis) 
class Param(Graph):
    def __init__(self, n, dol=None, basis=None, tree=None):
        super().__init__(n, dol)
        if basis is None:
            basis = []
            for idx in list(combinations(range(n), 2)): 
                g = Graph(n)
                g.AddEdge(idx[0], idx[1])
                basis.append(g.copy())
        self._basis = basis # list of DOLs
        self._basis_active = np.zeros(len(basis), dtype=bool) 
        self._tree = tree
    
    def GetBasis(self):
        return self._basis 
    
    def GetActiveBasis(self): 
        return self._basis_active
    
    def SetBasis(self, basis):
        self._basis = basis
    
    def BinAddOneBasis(self, idx): 
        self._basis_active[idx] = not self._basis_active[idx]
        self.BinaryAdd(self._basis[idx])
    
    def BinAddBasis(self, idx_l): 
        for i in idx_l: 
            self.BinAddOneBasis(i)
    
    def __repr__(self):
        #", \nbasis: " + self._basis.__str__()
        return "G: " + self._dol.__str__() + ", \nactive: " + self._basis_active.__str__() +\
                    ", \ntree:" + self._tree.__str__()
    def copy(self): 
        return copy.deepcopy(self)
    def __name__(self): 
        return "Graph and Basis"
        