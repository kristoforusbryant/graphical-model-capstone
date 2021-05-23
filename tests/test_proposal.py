from dists.Priors import BasisCount
from dists.Params import GraphAndBasis
from dists.CountPriors import TruncatedNB
from dists.TreePriors import Uniform
from dists.Proposals import BasisWalk
from utils.Graph import Graph
import numpy as np


def in_cycle_space(g):
    return np.all(list(map(lambda l: len(l) % 2 == 0, g._dol.values())))

# def g_is_sum_of_basis(g):
#     temp = Graph(len(g))
#     for i in range(len(g._basis)):
#         if g._basis_active[i]:
#             GraphAndBasis.BinaryAdd(temp, g._basis[i])
#     return temp == g

def g_is_sum_of_basis(g):
    sob = np.sum(g._basis.transpose()[g._basis_active], axis = 0)
    return (sob == g.GetBinaryL()).all()

def one_edge_difference(g, g_):
    x = np.sum(np.array(g._basis_active).astype(int) -
               np.array(g_._basis_active).astype(int))
    return (np.abs(x) == 1)

def test_proposal_Sample_no_COB():
    n = 10
    ct_prior = TruncatedNB(6, .75)
    prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))
    prop = BasisWalk(n, GraphAndBasis, Uniform(n))
    np.random.seed(123)
    for _ in range(20):
        g = prior.Sample()
        g_ = prop.Sample(g.copy())

        assert(g._tree == g_._tree)
        assert(one_edge_difference(g, g_))
        assert(in_cycle_space(g_))
        assert(g_is_sum_of_basis(g_))


