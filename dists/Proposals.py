import numpy as np
from utils.generate_basis import cycle_basis, COBM
from utils.inverse_binary import inverse_GF2, GF2


class BasisWalk:
    def __init__(self, n, Param, tree_prior=None, skip=None):
        # rv_size: distribution of the basis size
        self._n = n
        self._Param = Param
        self.tree_prior = tree_prior
        self.counter = 0
        self._skip = skip
        self._temp = None # to store (part of) previous parameter

    __name__ = 'basis_size_with_tree_prior'

    def Sample(self, param):
        # When basis changes
        if self._skip is not None:
            if self.counter % self._skip != 0:
                param_ = param
            else:
                M = COBM(param._tree)
                self._temp = param.copy()
                param_ = param

                # Make one switch first
                i = np.random.choice(range(len(param_._basis)))
                param_.BinAddOneBasis(i)

                # Sample tree that generates a new basis (assume T is uniform, hence does not affect proposal)
                T_ = self._tree_prior.Sample()

                param_._basis = cycle_basis(T_)
                param_._tree = T_

                M_ = COBM(param_._tree) # M and M_ are COBM in the entire space
                subM = (inverse_GF2(M_) @ GF2(M) % 2)[:len(param_._basis), :len(param_._basis)]
                _basis_active = (subM @ param_._basis_active % 2)
                param_._basis_active = _basis_active.astype(bool)

            self.counter += 1

            # Same procedure as basis_size.py
            i = np.random.choice(range(len(param_._basis)))
            param_.BinAddOneBasis(i)
            self._temp = i

        # When basis stays constant
        else:
            param_ = param
            # Same procedure as basis_size.py
            i = np.random.choice(range(len(param_._basis)))
            param_.BinAddOneBasis(i)
            self._temp = i

        return param_

    def Revert(self,p_):
        if self._skip is None:
            p_.BinAddOneBasis(self._temp)
            return p_
        elif self.counter % self._skip != 0:
            p_.BinAddOneBasis(self._temp)
            return p_
        else:
            return self._temp


    def PDF(self, p, p_):
        return 0

    def ParamType(self):
        return self._Param.__name__