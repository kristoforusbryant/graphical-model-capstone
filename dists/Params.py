from utils.Graph import Graph
import numpy as np
import copy
from utils.generate_basis import edge_basis, cycle_basis
import galois
GF2 = galois.GF(2)

# (Graph, Basis, Active Basis)
class GraphAndBasis(Graph):
    def __init__(self, n, dol=None, tree=None, basis=None):
        super().__init__(n, dol)
        self._tree = tree
        # basis is n_edges x n_basis array
        if tree:
            self._basis = cycle_basis(tree)
        elif basis is not None:
            self._basis = basis
        else:
            self._basis = edge_basis(n)
        self._basis_active = GF2(np.zeros(self._basis.shape[1], dtype=int))

    def GetBasis(self):
        return self._basis

    def GetActiveBasis(self):
        return self._basis_active

    def SetBasis(self, basis):
        self._basis = basis

    def _graph_from_binarr(self, n, a):
        triu_l = np.vstack(np.triu_indices(n,1)).transpose()
        dol = {i:[] for i in range(n)}
        for i, j in triu_l[np.where(a)[0]]:
            dol[i].append(j)
            dol[j].append(i)
        return Graph(n, dol)

    def BinAddOneBasis(self, idx):
        self._basis_active[idx] += GF2(1)
        b = self._graph_from_binarr(len(self), self._basis.transpose()[idx])
        self.BinaryAdd(b)

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
