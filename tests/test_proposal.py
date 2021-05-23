from dists.Priors import BasisCount
from dists.Params import GraphAndBasis
from dists.CountPriors import TruncatedNB
from dists.TreePriors import Uniform
from dists.Proposals import BasisWalk
from utils.Graph import Graph
import numpy as np


def in_cycle_space(g):
    return np.all(list(map(lambda l: len(l) % 2 == 0, g._dol.values())))

def g_is_sum_of_basis(g):
    sob = np.sum(g._basis.transpose()[np.where(g._basis_active)], axis = 0)
    return (sob == g.GetBinaryL()).all()

def one_edge_difference(g, g_):
    x = np.sum(np.array(g._basis_active).astype(int) -
               np.array(g_._basis_active).astype(int))
    return (np.abs(x) == 1)

def test_proposal_Sample_no_COB():
    n = 10
    ct_prior = TruncatedNB(n, .75)
    prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))
    prop = BasisWalk(n, GraphAndBasis, Uniform(n))
    np.random.seed(123)
    for _ in range(50):
        g = prior.Sample()
        g_ = prop.Sample(g.copy())

        print(prop._temp)
        assert(isinstance(prop._temp, np.integer))
        assert(g._tree == g_._tree)
        assert(one_edge_difference(g, g_))
        assert(in_cycle_space(g_))
        assert(g_is_sum_of_basis(g_))


def test_proposal_Sample_with_COB():
    n = 10
    cob_freq = 2
    ct_prior = TruncatedNB(n, .75)
    prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))
    prop = BasisWalk(n, GraphAndBasis, Uniform(n), skip = cob_freq)
    np.random.seed(123)

    g = prior.Sample()
    for _ in range(100):
        if prop.counter % cob_freq == 0:
            g_ = prop.Sample(g.copy())
            assert(isinstance(prop._temp, GraphAndBasis))
            assert(in_cycle_space(g_))
            assert(g_is_sum_of_basis(g_))
            g = g_

        else:
            g_ = prop.Sample(g.copy())
            assert(isinstance(prop._temp, np.integer))
            assert(g._tree == g_._tree)
            assert(one_edge_difference(g, g_))
            assert(in_cycle_space(g_))
            assert(g_is_sum_of_basis(g_))
            g = g_
