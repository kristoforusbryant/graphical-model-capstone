from dists.Priors import BasisCount
from dists.Params import GraphAndBasis
from dists.CountPriors import TruncatedNB
from dists.TreePriors import Uniform
from utils.Graph import Graph
import numpy as np


def in_cycle_space(g):
    return np.all(list(map(lambda l: len(l) % 2 == 0, g._dol.values())))

def g_is_sum_of_basis(g):
    sob = np.sum(g._basis.transpose()[np.where(g._basis_active)], axis = 0)
    return (sob == g.GetBinaryL()).all()

def test_prior_Sample():
    n = 10
    ct_prior = TruncatedNB(n, .75)
    prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))
    np.random.seed(123)
    for _ in range(20):
        g = prior.Sample()
        assert(in_cycle_space(g))
        assert(g_is_sum_of_basis(g))
