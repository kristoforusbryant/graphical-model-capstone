import numpy as np
from utils.generate_basis import cycle_basis
# utils should be inside the path of the file that imports this

class Prior:
    def __init__(self, n, Param, prob_c, tree_prior=None):
        # prob_c: distribution of the number of bases i.e. "count"
        self._n = n
        self._Param = Param
        self._prob_c = prob_c # for 0, ... number of basis
        if tree_prior:
            self._tree_prior = tree_prior
        else:
            from dists.spanning_tree_priors.uniform import STPrior
            self._tree_prior = STPrior(n)

    __name__ = 'basis_count_size'

    def Sample(self):
        # Sample tree that generates a new basis (assume T is uniform, hence does not affect proposal)
        T_ = self._tree_prior.Sample()
        param = self._Param(self._n, basis=cycle_basis(T_), tree=T_)

        # the count is distributed as p_c
        unscaled_p_c = np.array([self._prob_c(k) for k in range(len(param._basis) + 1)])
        p_c = unscaled_p_c / np.sum(unscaled_p_c)
        count = np.random.choice(range(len(param._basis) + 1), p=p_c)

        # the sizes of the bases are distributed as p_s
        unscaled_p_s = np.array([self._prob_s(b.EdgeCount()) for b in param._basis])
        p_s = unscaled_p_s / np.sum(unscaled_p_s)
        idx = np.random.choice(np.arange(len(p_s)), p=p_s, size=count, replace=False)
        for i in idx:
            param.BinAddOneBasis(i)

        return param


    # This PDF is unnormalised over the span of the basis
    def PDF(self, param):
        log_p_c = np.log(self._prob_c(param.EdgeCount()))
        log_p_s = 0
        for i in range(len(param._basis)):
            if param._basis_active[i]:
                k = param._basis[i].EdgeCount()
                log_p_s += np.log(self._prob_s(k))
        return log_p_c + log_p_s

    def ParamType(self):
        return self._Param.__name__