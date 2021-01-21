import numpy as np
import networkx as nx
from utils.Graph import Graph
from utils.exact_decomposable import IG_decomposable
from utils.atay_kayis_MC import Atay_Kayis_MC
from utils.laplace_approximation import laplace_approx

class G_Wishart: 
    # Let this be 'friend' of Graph 
    def __init__(self, G, delta, D):
        assert len(G._dol) == D.shape[0], "different dimensions between G and D"
        self.G = G
        self._delta = delta 
        self._D = D
    
    # Computation of Normalising Factor
    def IG_Exact(self): 
        return IG_decomposable(self.G._dol, self._delta, self._D)
        
    def IG_MC(self): 
        # (Atay-Kayis & Massam, 2005)
        return Atay_Kayis_MC(self.G._dol, self._delta, self._D)
        
    def IG_LA(self):
        # (Piccioni, 2000)
        return laplace_approx(self.G._dol, self._delta, self._D)    