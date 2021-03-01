import numpy as np
from scipy.stats import nbinom
from itertools import combinations
from utils.generate_basis import cycle_basis, COBM
from utils.inverse_binary import inverse_binary

# Assume basis remain constant
class Proposal:
    def __init__(self, n, Param, prob_s, tree_prior=None, skip=10):
        # rv_size: distribution of the basis size
        self._n = n          
        self._Param = Param
        self._prob_s = prob_s
        if tree_prior:
            self._tree_prior = tree_prior
        else: 
            from dists.spanning_tree_priors.uniform import STPrior
            self._tree_prior = STPrior(n) 
        self.counter = 0 
        self._skip = skip
            
    __name__ = 'basis_size_with_tree_prior'
    
    def Sample(self, param):
        param_ = param.copy()
        
        # Change Basis
        if self.counter % self._skip == 0: 
            # Sample tree that generates a new basis (assume T is uniform, hence does not affect proposal)
            T_ = self._tree_prior.Sample()
    
            param_._basis = cycle_basis(T_)
            param_._tree = T_

            M = COBM(param._tree) 
            M_ = COBM(param_._tree) # M and M_ are COBM in the entire space
            subM = (inverse_binary(M_) @ M % 2)[:len(param_._basis), :len(param_._basis)]
            _basis_active = (subM @ param._basis_active % 2) 
            param_._basis_active = _basis_active.astype(bool)
        self.counter += 1 
        
        # Rescaling p_s 
        unscaled_p_s = np.array([self._prob_s(p.EdgeCount()) for p in param_._basis])
        self._p_s = unscaled_p_s / np.sum(unscaled_p_s)
        
        # Same procedure as basis_size.py
        i = np.random.choice(range(len(param_._basis)), p = self._p_s)
        param_.BinAddOneBasis(i)
        return param_ 
    
    def PDF(self, p0, p1):
        return 0 
    def ParamType(self): 
        return self._Param.__name__