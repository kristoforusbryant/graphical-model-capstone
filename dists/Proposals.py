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

    __name__ = 'basis_size_with_tree_moves'

    def Sample(self, param):
        # When basis can change
        if self._skip is not None:
            if self.counter % self._skip == 0:
                # too much of a hassle to revert by adding bases, just add the entire object
                self._temp = param.copy()

                param_ = param

                # change basis
                T_ = self._tree_prior.Sample()
                while param._tree == T_:
                    T_ = self._tree_prior.Sample()
                param_ = change_basis(param_, T_)

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

    def PDF_ratio(self, p_):
        return 0

    def PDF(self, p, p_):
        return 0

    def ParamType(self):
        return self._Param.__name__


class BasisBD:
    def __init__(self, n, Param, tree_prior=None, skip=None, alpha=.5):
        # rv_size: distribution of the basis size
        self._n = n
        self._Param = Param
        self._tree_prior = tree_prior
        self.counter = 0
        self._skip = skip
        self._temp = None # to store (part of) previous parameter
        self._alpha = alpha # probability of birth

    __name__ = 'basis_BD_with_tree_moves'

    def Sample(self, param):
        # When basis can change
        if self._skip is not None:
            if self.counter % self._skip == 0:
                # too much of a hassle to revert by adding bases, just add the entire object
                self._temp = param.copy()

                param_ = param

                # change basis
                T_ = self._tree_prior.Sample()
                while param._tree == T_:
                    T_ = self._tree_prior.Sample()
                param_ = change_basis(param_, T_)

            else:
                param_ = param
                if np.sum(param_._basis_active == 1) == 0: # empty, hence birth
                    idx = np.where(param_._basis_active == 0)[0]
                    i = np.random.choice(idx)
                    param_.BinAddOneBasis(i)
                    self._temp = i
                elif np.sum(param_._basis_active == 0) == 0: # complete, hence death
                    idx = np.where(param_._basis_active == 1)[0]
                    i = np.random.choice(idx)
                    param_.BinAddOneBasis(i)
                    self._temp = i
                elif np.random.random() < self._alpha : # birth
                    idx = np.where(param_._basis_active == 0)[0]
                    i = np.random.choice(idx)
                    param_.BinAddOneBasis(i)
                    self._temp = i
                else: # death
                    idx = np.where(param_._basis_active == 1)[0]
                    i = np.random.choice(idx)
                    param_.BinAddOneBasis(i)
                    self._temp = i

        # When basis stays constant
        else:
            param_ = param
            if np.sum(param_._basis_active == 1) == 0: # empty, hence birth
                idx = np.where(param_._basis_active == 0)[0]
                i = np.random.choice(idx)
                param_.BinAddOneBasis(i)
                self._temp = i
            elif np.sum(param_._basis_active == 0) == 0: # complete, hence death
                idx = np.where(param_._basis_active == 1)[0]
                i = np.random.choice(idx)
                param_.BinAddOneBasis(i)
                self._temp = i
            elif np.random.random() < self._alpha : # birth
                idx = np.where(param_._basis_active == 0)[0]
                i = np.random.choice(idx)
                param_.BinAddOneBasis(i)
                self._temp = i
            else: # death
                idx = np.where(param_._basis_active == 1)[0]
                i = np.random.choice(idx)
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

    def _get_pdf(self, n_active, n_active_, n_bases): # q(p -> p_)
        if n_active == 0 or n_active == n_bases: # guaranteed birth or death, respectively
            return np.log(1) - np.log(n_bases)
        elif n_active_ > n_active: # birth process
            return np.log(self._alpha) - np.log(n_bases - n_active)
        else: #death process
            return np.log(1 - self._alpha) - np.log(n_active)

    def PDF_ratio(self, p_): # q(p_ -> p) / q(p -> p_)
        if self._skip is None or ((self.counter - 1) % self._skip != 0):
            n_active_ = np.sum(np.array(p_._basis_active, dtype=int))
            if self._temp in np.where(p_._basis_active == 1)[0]:
                n_active = n_active_ - 1
            else:
                n_active = n_active_ + 1
            n_bases = len(p_._basis_active)

            return self._get_pdf(n_active_, n_active, n_bases) - self._get_pdf(n_active, n_active_, n_bases)
        else:
            return 0

    # def PDF_ratio(self, p_): # q(p_ -> p) / q(p -> p_)
    #     if self._skip is None or ((self.counter - 1) % self._skip != 0):
    #         n_active_ = np.sum(np.array(p_._basis_active, dtype=int))
    #         if n_active_ == 0: # p_ must be the result of death
    #             n_active = n_active_ + 1
    #             return np.log(1) - np.log(len(p_._basis_active) - n_active_) - (np.log(1 - self._alpha) - np.log(n_active))
    #         elif n_active_ == len(p_._basis_active): # p_ must be the result of birth
    #             n_active = n_active_ - 1
    #             return np.log(1) - np.log(n_active_) - (np.log(self._alpha) - np.log(len(p_._basis_active) - n_active))
    #         elif self._temp in np.where(p_._basis_active == 1)[0]: # p_ is result of birth, therefore p_ -> p is a death process
    #             n_active = n_active_ - 1
    #             return np.log(1 - self._alpha) - np.log(n_active_) - (np.log(self._alpha) - np.log(len(p_._basis_active) - n_active))
    #         else: # p_ is result of death, therefore p_ -> p is a birth process
    #             n_active = n_active_ + 1
    #             return np.log(self._alpha) - np.log(len(p_._basis_active) - n_active_) - (np.log(1 - self._alpha) - np.log(n_active))
    #     else:
    #         return 0

    def PDF(self, p, p_):
        return 0

    def ParamType(self):
        return self._Param.__name__