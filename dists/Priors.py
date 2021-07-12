import numpy as np
from utils.generate_basis import cycle_basis, edge_basis
from scipy import special

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
            # Sample tree that generates a new basis (assume T is uniform)
            T_ = self._tree_prior.Sample()
            param = self._Param(self._n, tree=T_)
        else:
            param = self._Param(self._n, basis=self._basis)

        # the count is distributed as p_c
        ln_p_c = np.array([self._prob_c(k) for k in range(param._basis.shape[1] + 1)])
        unscaled_p_c = np.exp(ln_p_c - np.max(ln_p_c))
        p_c = unscaled_p_c / np.sum(unscaled_p_c)
        count = np.random.choice(range(param._basis.shape[1] + 1), p=p_c)

        # the sizes of the bases are distributed as p_s
        idx = np.random.choice(range(param._basis.shape[1]), size=count, replace=False)
        for i in idx:
            param.BinAddOneBasis(i)

        return param

    def _chooseln(self, N, k):
        return special.gammaln(N+1) - special.gammaln(N-k+1) - special.gammaln(k+1)

    def PDF(self, param):
        size = np.sum(np.array(param._basis_active, dtype=int))
        total_edges = len(param._basis_active)
        return self._prob_c(size) - self._chooseln(total_edges, size)

    def ParamType(self):
        return self._Param.__name__

class BasisInclusion:
    def __init__(self, n, Param, alpha=.5, tree_prior=None, basis=None):
        # alpha: probability of inclusion
        self._n = n
        self._Param = Param
        self._alpha = alpha
        if tree_prior:
            self._tree_prior = tree_prior
        elif basis is not None:
            self._basis = basis
            self._tree_prior = None
        else:
            self._basis = edge_basis(n)
            self._tree_prior = None

    __name__ = 'basis_inclusion_prob'

    def Sample(self):
        if self._tree_prior:
            # Sample tree that generates a new basis (assume T is uniform)
            T_ = self._tree_prior.Sample()
            param = self._Param(self._n, tree=T_)
        else:
            param = self._Param(self._n, basis=self._basis)

        idx = np.where(np.random.random(param._basis.shape[1]) < self._alpha)[0]
        param.BinAddBasis(idx)

        return param

    def PDF(self, param):
        n_active = np.sum(np.array(param._basis_active, dtype=int))
        return np.log(self._alpha) * n_active + np.log(1 - self._alpha) * (len(param._basis_active) - n_active)

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