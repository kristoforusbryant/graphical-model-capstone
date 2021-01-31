from utils.Graph import Graph

# (Graph, Basis, Active Basis) 
class Param(Graph):
    def __init__(self, n, dol=None, basis=None):
        super().__init__(n, dol)
        self._basis = basis # list of DOLs
        