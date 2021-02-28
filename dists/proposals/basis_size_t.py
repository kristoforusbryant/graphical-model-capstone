import numpy as np
from scipy.stats import nbinom
from itertools import combinations
from utils.generate_basis import cycle_basis

# Assume basis remain constant
class Proposal:
    def __init__(self, n, Param, basis, prob_s, tree_prior=None):
        # rv_size: distribution of the basis size
        self._n = n          
        self._Param = Param
        self._basis = basis
        self._prob_s = prob_s
        unscaled_p_s = np.array([self._prob_s(p.EdgeCount()) for p in self._basis])
        self._p_s = unscaled_p_s / np.sum(unscaled_p_s)
        if tree_prior:
            self._tree_prior = tree_prior
        else: 
            from dists.spanning_tree_priors.uniform import STPrior
            self._tree_prior = STPrior(n) 
            
    __name__ = 'basis_size_with_tree_prior'
    
    def Sample(self, param):
        # Sample tree that generates a new basis (assume T is uniform, hence does not affect proposal)
        T = self._tree_prior.Sample()
        self._basis = cycle_basis(T)
        
        # Same procedure as basis_size.py
        i = np.random.choice(range(len(param._basis)), p = self._p_s)
        param_ = param.copy()
        param_.BinAddOneBasis(i)
        return param_ 
    
    def PDF(self, p0, p1):
        return 0 
    def ParamType(self): 
        return self._Param.__name__