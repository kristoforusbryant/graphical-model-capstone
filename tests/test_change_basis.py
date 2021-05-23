from dists.Priors import BasisCount
from dists.Params import GraphAndBasis
from dists.CountPriors import TruncatedNB
from dists.TreePriors import Uniform
import numpy as np

from utils.generate_basis import cycle_basis_complete, cycle_basis, change_basis
from utils.inverse_binary import GF2


def g_is_sum_of_basis(g):
    sob = np.sum(g._basis.transpose()[np.where(g._basis_active)], axis = 0)
    return (sob == g.GetBinaryL()).all()

def test_cycle_basis_complete():
    n = 20
    treeprior = Uniform(n)
    np.random.seed(123)
    for _ in range(50):
        t = treeprior.Sample()
        m = n * (n - 1) // 2
        assert(np.linalg.matrix_rank(GF2(cycle_basis_complete(t))) == m)

def test_change_basis():
    n = 20
    ct_prior = TruncatedNB(n, .75)
    tree_prior = Uniform(n)
    prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))

    np.random.seed(123)
    for _ in range(5):
        g = prior.Sample()
        t =  tree_prior.Sample()
        g_ = change_basis(g.copy(), t)

        assert(g_ == g)
        assert(g_._tree == t)
        assert((g_._basis == cycle_basis(t)).all())
        assert(g_is_sum_of_basis(g_))