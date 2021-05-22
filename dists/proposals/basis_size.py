import numpy as np

# Assume basis remain constant
class Proposal:
    def __init__(self, n, Param, basis, prob_s):
        # rv_size: distribution of the basis size
        self._n = n
        self._Param = Param
        self._basis = basis
        self._prob_s = prob_s
        unscaled_p_s = np.array([self._prob_s(p.EdgeCount()) for p in self._basis])
        self._p_s = unscaled_p_s / np.sum(unscaled_p_s)
    __name__ = 'basis_size'

    def Sample(self, param):
        i = np.random.choice(range(len(param._basis)), p = self._p_s)
        param_ = param.copy()
        param_.BinAddOneBasis(i)
        return param_

    def PDF(self, p0, p1):
        return 0
    def ParamType(self):
        return self._Param.__name__