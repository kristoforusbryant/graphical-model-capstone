# Birth-Death Proposal 
import numpy as np
from scipy.stats import nbinom
from itertools import combinations
from utils.generate_basis import cycle_basis
from copy import deepcopy
from utils.generate_basis import cycle_basis, COBM
from utils.inverse_binary import inverse_binary

# Assume basis remain constant
class Proposal:
    def __init__(self, n, Param, tree_prior=None, bd_prob=.5, skip=10):
        # prob_s: distribution of the basis size
        self._n = n          
        self._Param = Param
        self._bd_prob = bd_prob
        if tree_prior:
            self._tree_prior = tree_prior
        else: 
            from dists.spanning_tree_priors.uniform import STPrior
            self._tree_prior = STPrior(n) 
        self.counter = 0 
        self._skip = skip
            
    __name__ = 'basis_size_bd_with_tree_prior'
    
    def Sample(self, param, with_switching=False):
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
        
        indexes = np.arange(len(param_._basis))
        # if empty, then guaranteed birth 
        if np.sum(param._basis_active) == 0: 
            # birth
            inactive_idx = np.invert(param._basis_active)
            i = np.random.choice(indexes[inactive_idx])
            param_ = param.copy()
            param_.BinAddOneBasis(i)
            
        # if complete, then guaranteed death 
        elif np.sum(param._basis_active) == len(param._basis_active): 
            # death 
            active_idx = param._basis_active
            i = np.random.choice(indexes[active_idx])
            param_ = param.copy()
            param_.BinAddOneBasis(i)
        
        # else flip a coin 
        elif np.random.random() < self._bd_prob: 
            # birth 
            inactive_idx = np.invert(param._basis_active)
            i = np.random.choice(indexes[inactive_idx])
            param_ = param.copy()
            param_.BinAddOneBasis(i)
            
        else: 
            # death
            active_idx = param._basis_active
            i = np.random.choice(indexes[active_idx])
            param_ = param.copy()
            param_.BinAddOneBasis(i)
            
        is_special = (np.sum(param_._basis_active) == 0 or np.sum(param_._basis_active) == len(param_._basis_active))
        if with_switching and not is_special: 
            i = np.random.choice(indexes[param_._basis_active])
            j = np.random.choice(indexes[np.invert(param_._basis_active)])
            param_.BinAddOneBasis(i)
            param_.BinAddOneBasis(j)
            
        return param_ 
    
    def PDF(self, p0, p1):
        # if empty or complete 
        if n_active_p0 == 0 or n_active_p0 == len(self._basis):
            return 0
        
        if n_active_p0 < n_active_p1:
            # birth
            inactive_idx = np.invert(p0._basis_active)            
            return np.log(self._bd_prob) 
        else: 
            # death 
            active_idx = p0._basis_active            
            return np.log(1 - self._bd_prob) 
        
    def ParamType(self): 
        return self._Param.__name__