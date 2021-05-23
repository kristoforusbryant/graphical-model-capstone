from utils.MCMC import MCMC_Sampler
import dists.TreePriors, dists.CountPriors
import dists.Priors, dists.Likelihoods, dists.Proposals
from dists.Params import GraphAndBasis
from utils.generate_basis import circle_graph, edge_basis
from tests.test_utils.generate_data import generate_data
import numpy as np
from utils.diagnostics import str_list_to_median_graph


# aux functions
def lookup_contains_all_params(sampler):
    return not (set(sampler.res['PARAMS']).union(set(sampler.res['SAMPLES'])) - set(sampler.lookup.keys()))

def only_propose_one_edge_difference(sampler):
    try:
        for i in range(len(sampler.res['SAMPLES'])):
            a = np.array(list(sampler.res['SAMPLES'][i])).astype(int)
            b = np.array(list(sampler.res['PARAMS'][i])).astype(int)
            diff = np.sum(np.abs(a - b))
            if sampler.res['ACCEPT_INDEX'][i] == 1:
                assert((a == b).all())
            else:
                assert(diff == 1)
    except AssertionError:
        False
    return True

def lookup_types_are_correct(n, lookup):
    try:
        for k, l in lookup.items():
            assert(len(k) == n * (n - 1) // 2)
            assert(isinstance(l['LIK'], float) and isinstance(l['PRIOR'], float))
            assert(len(l['BASIS_ID']) == (n - 1) * (n - 2) //2
                    and len(l['TREE_ID']) == n * (n - 1) // 2)
    except AssertionError:
        False
    return True

def run_MCMC(g, n, n_obs, tree_prior, ct_prior, basis=None, skip=None, seed=123, fixed_init=False):
    data = generate_data(n, n_obs, g, seed=123)
    prior = dists.Priors.BasisCount(n, GraphAndBasis, ct_prior, tree_prior, basis)
    lik = dists.Likelihoods.GW(data, 3, np.eye(n), GraphAndBasis)
    prop = dists.Proposals.BasisWalk(n, GraphAndBasis, tree_prior, skip=skip)

    sampler = MCMC_Sampler(prior, prop, lik, data)
    np.random.seed(seed)
    if fixed_init:
        sampler.run(1500, g)
    else:
        sampler.run(1500)
    return sampler


def test_MCMC_edge_0():
    n, n_obs = (5, 1000)
    basis = edge_basis(n)
    tree_prior = None
    ct_prior = dists.CountPriors.Uniform()

    # Empty
    g = GraphAndBasis(n)
    sampler = run_MCMC(g, n, n_obs, tree_prior, ct_prior, basis=basis, seed=123)
    print(sampler.res['LIK_'][-1])
    median_graph = str_list_to_median_graph(n, sampler.res['SAMPLES'][500:])

    assert((median_graph == g.GetAdjM()).all())
    assert(only_propose_one_edge_difference(sampler))
    assert(lookup_contains_all_params(sampler))

    # Circle
    g = GraphAndBasis(n, circle_graph(n))
    sampler = run_MCMC(g, n, n_obs, tree_prior, ct_prior, basis=basis, seed=123)
    print(sampler.res['LIK_'][-1])
    median_graph = str_list_to_median_graph(n, sampler.res['SAMPLES'][500:], .75)

    assert((median_graph == g.GetAdjM()).all())
    assert(only_propose_one_edge_difference(sampler))
    assert(lookup_contains_all_params(sampler))

"""
{0: [1, 2, 4], 1: [0, 2, 3], 2: [0, 1, 3, 4], 3: [1, 2], 4: [0, 2]}
loglik: -3703.821170587811, logprior: 0.0
-3679.316506725288
{0: [1, 2, 4], 1: [0, 2, 3], 2: [0, 1, 3, 4], 3: [1, 2], 4: [0, 2]}
loglik: -793.8125859817117, logprior: 0.0
-920.7841603658132
"""


def test_MCMC_edge_1():
    n, n_obs = (5, 1000)
    basis = edge_basis(n)
    tree_prior = None
    ct_prior = dists.CountPriors.Uniform()

    np.random.seed(123)
    for i in range(3):
        g = GraphAndBasis(n)
        g.SetRandom()
        sampler = run_MCMC(g, n, n_obs, tree_prior, ct_prior, basis=basis, seed=123, fixed_init=True)
        print(sampler.res['LIK_'][-1])
        median_graph = str_list_to_median_graph(n, sampler.res['SAMPLES'][500:], .5)

        assert(np.sum(median_graph != g.GetAdjM()) < 3)
        assert(only_propose_one_edge_difference(sampler))
        assert(lookup_contains_all_params(sampler))
"""
{0: [2], 1: [], 2: [0, 3, 4], 3: [2], 4: [2]}
loglik: -1957.4570827773334, logprior: 0.0
-2528.64373384143

{0: [2], 1: [2, 4], 2: [0, 1, 3], 3: [2], 4: [1]}
loglik: -66.67734578275466, logprior: 0.0
-68.72425985733959

{0: [3, 4], 1: [2, 4], 2: [1], 3: [0, 4], 4: [0, 1, 3]}
loglik: 1906.0051120391747, logprior: 0.0
1906.0051120391747
"""

def only_propose_one_basis_difference(sampler, skip):
    try:
        for i in range(len(sampler.res['SAMPLES'])):
            if i % skip != 0:
                b_str1 = sampler.lookup[sampler.res['SAMPLES'][i]]['BASIS_ID']
                b_str2 = sampler.lookup[sampler.res['PARAMS'][i]]['BASIS_ID']
                a = np.array(list(b_str1)).astype(int)
                b = np.array(list(b_str2)).astype(int)
                diff = np.sum(np.abs(a - b))
                if sampler.res['ACCEPT_INDEX'][i] == 1:
                    assert((a == b).all())
                else:
                    assert(diff == 1)
    except AssertionError:
        False
    return True


def test_MCMC_cb_0():
    n, n_obs = (5, 1000)
    tree_prior = dists.TreePriors.Uniform(n)
    ct_prior = dists.CountPriors.TruncatedNB(n, .75)
    cob_freq = 2

    # Empty
    g = GraphAndBasis(n)
    sampler = run_MCMC(g, n, n_obs, tree_prior, ct_prior, skip=cob_freq, seed=123)
    print(sampler.res['LIK_'][-1])
    median_graph = str_list_to_median_graph(n, sampler.res['SAMPLES'][500:])

    assert((median_graph == g.GetAdjM()).all())
    assert(only_propose_one_basis_difference(sampler, cob_freq))
    assert(lookup_contains_all_params(sampler))

    # Circle
    g = GraphAndBasis(n, circle_graph(n))
    sampler = run_MCMC(g, n, n_obs, tree_prior, ct_prior, skip=cob_freq, seed=123)
    print(sampler.res['LIK_'][-1])
    median_graph = str_list_to_median_graph(n, sampler.res['SAMPLES'][500:])

    assert((median_graph == g.GetAdjM()).all())
    assert(only_propose_one_basis_difference(sampler, cob_freq))
    assert(lookup_contains_all_params(sampler))

"""
{0: [2, 3], 1: [], 2: [0, 3], 3: [0, 2], 4: []}
loglik: -3686.7067620458174, logprior: -1.215266810944695
-3686.7067620458174
{0: [2, 3], 1: [], 2: [0, 3], 3: [0, 2], 4: []}
loglik: -2200.310533794313, logprior: -1.215266810944695
-946.533059800595
"""

def if_not_similar_then_median_has_larger_lik(g, median_graph, sampler):
    triu = np.triu_indices(len(g), 1)
    median_is_larger = sampler.lookup[''.join(median_graph[triu].astype(str))]['LIK'] > \
                       sampler.lookup[g.GetID()]['LIK']
    return (np.sum(median_graph != g.GetAdjM()) < 3) or median_is_larger

def test_MCMC_cb_1():
    n, n_obs = (5, 1000)
    tree_prior = dists.TreePriors.Uniform(n)
    ct_prior = dists.CountPriors.TruncatedNB(n, .75)
    cob_freq = 2

    np.random.seed(123)
    for _ in range(3):
        prior = dists.Priors.BasisCount(n, GraphAndBasis, ct_prior, tree_prior)
        g = prior.Sample()

        data = generate_data(n, n_obs, g, seed=123)
        lik = dists.Likelihoods.GW_LA(data, 3, np.eye(n), GraphAndBasis)
        prop = dists.Proposals.BasisWalk(n, GraphAndBasis, tree_prior, skip=cob_freq)

        sampler = MCMC_Sampler(prior, prop, lik, data)

        np.random.seed(123)
        sampler.run(1500, g)

        print(sampler.res['LIK_'][-1])
        median_graph = str_list_to_median_graph(n, sampler.res['SAMPLES'][500:], .5)

        assert(only_propose_one_basis_difference(sampler, cob_freq))
        assert(lookup_contains_all_params(sampler))
        assert(lookup_types_are_correct(n, sampler.lookup))
        assert(if_not_similar_then_median_has_larger_lik(g, median_graph, sampler))

