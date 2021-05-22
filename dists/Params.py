from utils.Graph import Graph
import numpy as np
import copy
from utils.generate_basis import edge_basis, cycle_basis

# (Graph, Basis, Active Basis)
class GraphAndBasis(Graph):
    def __init__(self, n, dol=None, tree=None, basis=None):
        super().__init__(n, dol)
        self._tree = tree
        if tree:
            self._basis = cycle_basis(tree)
        elif basis:
            self._basis = basis
        else:
            self._basis = edge_basis(n)
        self._basis_active = np.zeros(len(self._basis), dtype=bool)

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
        return "G: " + self._dol.__str__() + ", \nactive: " + self._basis_active.__str__() +\
                    ", \ntree:" + self._tree.__str__()

    def copy(self):
        return copy.deepcopy(self)

    def __name__(self):
        return "Graph and Basis"
