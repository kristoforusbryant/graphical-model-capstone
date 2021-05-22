from dists.Priors import BasisCount
from dists.Params import GraphAndBasis
from dists.CountPriors import TruncatedNB
from dists.TreePriors import Uniform
from utils.Graph import Graph
import numpy as np


def in_cycle_space(g):
    return np.all(list(map(lambda l: len(l) % 2 == 0, g._dol.values())))

def g_is_sum_of_basis(g):
    temp = Graph(len(g))
    for i in range(len(g._basis)):
        if g._basis_active[i]:
            GraphAndBasis.BinaryAdd(temp, g._basis[i])
    return temp == g

def test_prior_Sample():
    n = 10
    ct_prior = TruncatedNB(6, .75)
    prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))
    np.random.seed(123)
    for _ in range(20):
        g = prior.Sample()
        assert(in_cycle_space(g))
        assert(g_is_sum_of_basis(g))
