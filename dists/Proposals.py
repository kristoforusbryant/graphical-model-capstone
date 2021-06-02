import numpy as np
from utils.generate_basis import change_basis

class BasisWalk:
    def __init__(self, n, Param, tree_prior=None, skip=None):
        # rv_size: distribution of the basis size
        self._n = n
        self._Param = Param
        self._tree_prior = tree_prior
        self.counter = 0
        self._skip = skip
        self._temp = None # to store (part of) previous parameter

    __name__ = 'basis_size_with_tree_prior'

    def Sample(self, param):
        # When basis can change
        if self._skip is not None:
            if self.counter % self._skip == 0:
                # too much of a hassle to revert by adding bases, just add the entire object
                self._temp = param.copy()

                param_ = param

                # move once
                # i = np.random.choice(range(param_._basis.shape[1]))
                # param_.BinAddOneBasis(i)

                # change basis
                T_ = self._tree_prior.Sample()
                param_ = change_basis(param_, T_)

                # move again
                # i = np.random.choice(range(param_._basis.shape[1]))
                # param_.BinAddOneBasis(i)
            else:
                param_ = param

                i = np.random.choice(range(param_._basis.shape[1]))
                param_.BinAddOneBasis(i)
                self._temp = i

        # When basis stays constant
        else:
            param_ = param
            # Same procedure as basis_size.py
            i = np.random.choice(range(param_._basis.shape[1]))
            param_.BinAddOneBasis(i)
            self._temp = i

        self.counter += 1

        return param_

    def Revert(self,p_):
        if self._skip is None:
            p_.BinAddOneBasis(self._temp)
            return p_
        elif (self.counter - 1) % self._skip != 0:
            p_.BinAddOneBasis(self._temp)
            return p_
        else:
            return self._temp

    def PDF(self, p, p_):
        return 0

    def ParamType(self):
        return self._Param.__name__