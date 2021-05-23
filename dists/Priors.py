import numpy as np
from utils.generate_basis import cycle_basis, edge_basis

class BasisCount:
    def __init__(self, n, Param, prob_c, tree_prior=None, basis=None):
        # prob_c: distribution of the number of bases i.e. "count"
        self._n = n
        self._Param = Param
        self._prob_c = prob_c # for 0, ... number of basis
        if tree_prior:
            self._tree_prior = tree_prior
        elif basis is not None:
            self._basis = basis
            self._tree_prior = None
        else:
            self._basis = edge_basis(n)
            self._tree_prior = None

    __name__ = 'basis_count_size'

    def Sample(self):
        if self._tree_prior:
            # Sample tree that generates a new basis (assume T is uniform, hence does not affect proposal)
            T_ = self._tree_prior.Sample()
            param = self._Param(self._n, basis=cycle_basis(T_), tree=T_)
        else:
            param = self._Param(self._n, basis=self._basis)

        # the count is distributed as p_c
        unscaled_p_c = np.array([self._prob_c(k) for k in range(param._basis.shape[1] + 1)])
        p_c = unscaled_p_c / np.sum(unscaled_p_c)
        count = np.random.choice(range(param._basis.shape[1] + 1), p=p_c)

        # the sizes of the bases are distributed as p_s
        idx = np.random.choice(range(param._basis.shape[1]), size=count, replace=False)
        for i in idx:
            param.BinAddOneBasis(i)

        return param

    def PDF(self, param):
        return np.log(self._prob_c(np.sum(param._basis_active)))

    def ParamType(self):
        return self._Param.__name__

class Uniform:
    def __init__(self, n, Param, basis=None):
        self._n = n
        self._Param = Param
        if basis is not None:
            self._basis = basis
        else:
            self._basis = edge_basis(n)

    __name__ = 'uniform'

    def Sample(self):
        param = self._Param(self._n)
        triu = np.triu_indices(self._n,1)
        for i, j in list(zip(triu[0], triu[1])):
            if np.random.uniform() > 0.5:
                    param.AddEdge(i,j)
        return param

    def PDF(self, param):
        return 0

    def ParamType(self):
        return self._Param.__name__