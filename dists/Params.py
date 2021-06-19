from utils.Graph import Graph
import numpy as np
import copy
from utils.generate_basis import cycle_basis_complete, edge_basis, cycle_basis
import galois
GF2 = galois.GF(2)

# (Graph, Basis, Active Basis)
class GraphAndBasis(Graph):
    def __init__(self, n, dol=None, tree=None, basis=None):
        super().__init__(n, dol)
        self._tree = tree
        # basis is n_edges x n_basis array
        if tree:
            cbc = cycle_basis_complete(tree)
            self._basis = cbc[:,:(n - 1) * (n - 2) // 2]
            self._basis_active = np.linalg.solve(cbc, GF2(self.GetBinaryL()))[:(n - 1) * (n - 2) // 2]
        elif basis is not None:
            assert(basis.shape == (n * (n - 1) // 2, n * (n - 1) // 2))
            self._basis = basis
            self._basis_active = np.linalg.solve(self._basis, GF2(self.GetBinaryL()))
        else:
            self._basis = edge_basis(n)
            self._basis_active = np.linalg.solve(self._basis, GF2(self.GetBinaryL()))

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

    def GetBasisNeighbours(self):
        l = [self.copy() for i in range(self._basis.shape[1])]
        for i in range(self._basis.shape[1]):
            l[i].BinAddOneBasis(i)
        return l

    def __repr__(self):
        return "G: " + self._dol.__str__() + ", \nactive: " + self._basis_active.__str__() +\
                    ", \ntree:" + self._tree.__str__()

    def copy(self):
        return copy.deepcopy(self)

    def __name__(self):
        return "Graph and Basis"
