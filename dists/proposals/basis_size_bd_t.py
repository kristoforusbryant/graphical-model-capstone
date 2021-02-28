# Birth-Death Proposal 
import numpy as np
from scipy.stats import nbinom
from itertools import combinations
from utils.generate_basis import cycle_basis
from copy import deepcopy

# Assume basis remain constant
class Proposal:
    def __init__(self, n, Param, basis, prob_s, tree_prior=None, bd_prob=.5):
        # prob_s: distribution of the basis size
        self._n = n          
        self._Param = Param
        self._basis = basis
        self._prob_s = prob_s
        unscaled_p_s = np.array([self._prob_s(p.EdgeCount()) for p in self._basis])
        self._p_s = unscaled_p_s / np.sum(unscaled_p_s) 
        self._bd_prob = bd_prob
        if tree_prior:
            self._tree_prior = tree_prior
        else: 
            from dists.spanning_tree_priors.uniform import STPrior
            self._tree_prior = STPrior(n) 
            
    __name__ = 'basis_size_bd_with_tree_prior'
    
    def Sample(self, param, with_switching=False):
        # sampling new basis 
        T = self._tree_prior.Sample()
        self._basis = cycle_basis(T)
        param._basis = deepcopy(self._basis)
        
        # switching basis
        A = np.zeros((len(self._basis), int(self._n * (self._n - 1)/2)) # change of basis matrix from param -> new 
        for i in range(len(self._basis)): 
            A[i,:] = (param._basis[i].GetBinaryL() + self._basis[i].GetBinaryL()) % 2
        
        param._basis_active = ((A @ param._basis_active) % 2).astype(bool).tolist()
        
        indexes = np.arange(len(self._basis))
        # if empty, then guaranteed birth 
        if np.sum(param._basis_active) == 0: 
            # birth
            inactive_idx = np.invert(param._basis_active)
            subset_p_s = self._p_s[inactive_idx] / np.sum(self._p_s[inactive_idx])
            i = np.random.choice(indexes[inactive_idx], p=subset_p_s)
            param_ = param.copy()
            param_.BinAddOneBasis(i)
            
        # if complete, then guaranteed death 
        elif np.sum(param._basis_active) == len(param._basis_active): 
            # death 
            active_idx = param._basis_active
            subset_p_s = self._p_s[active_idx] / np.sum(self._p_s[active_idx])
            i = np.random.choice(indexes[active_idx], p=subset_p_s)
            param_ = param.copy()
            param_.BinAddOneBasis(i)
        
        # else flip a coin 
        elif np.random.random() < self._bd_prob: 
            # birth 
            inactive_idx = np.invert(param._basis_active)
            subset_p_s = self._p_s[inactive_idx] / np.sum(self._p_s[inactive_idx])
            i = np.random.choice(indexes[inactive_idx], p=subset_p_s)
            param_ = param.copy()
            param_.BinAddOneBasis(i)
            
        else: 
            # death
            active_idx = param._basis_active
            subset_p_s = self._p_s[active_idx] / np.sum(self._p_s[active_idx])
            i = np.random.choice(indexes[active_idx], p=subset_p_s)
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
        # Note that the same-dimensional switching cancels out in the ratio, hence not included  
        indexes = np.arange(len(self._basis))
        n_active_p0 = np.sum(p0._basis_active)
        n_active_p1 = np.sum(p1._basis_active)
        flipped = indexes[np.bitwise_xor(p0._basis_active, p1._basis_active)][0]
        
        if n_active_p0 == n_active_p1: 
            raise ValueError("p0 and p1 has the same number of active basis")
        
        # if empty or complete 
        if n_active_p0 == 0 or n_active_p0 == len(self._basis):
            return np.log(self._p_s[flipped])
        
        if n_active_p0 < n_active_p1:
            # birth
            inactive_idx = np.invert(p0._basis_active)            
            return np.log(self._bd_prob) + np.log(self._p_s[flipped] / np.sum(self._p_s[inactive_idx]))
        else: 
            # death 
            active_idx = p0._basis_active            
            return np.log(1 - self._bd_prob) + np.log(self._p_s[flipped] / np.sum(self._p_s[active_idx]))
        
    def ParamType(self): 
        return self._Param.__name__