from utils.MCMC import MCMC_Sampler
import dists.TreePriors, dists.CountPriors
import dists.Priors, dists.Likelihoods, dists.Proposals
from dists.Params import GraphAndBasis, Graph
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

def sampler_types_are_correct(n, sampler):
    try:
        for k, l in sampler.lookup.items():
            assert(len(k) == n * (n - 1) // 2)
            assert(isinstance(l['LIK'], float))

        for i in range(sampler.iter):
            assert(isinstance(sampler.res['PARAMS_PROPS'][i]['PRIOR'], float))
            assert(len(sampler.res['PARAMS_PROPS'][i]['BASIS_ID']) == (n - 1) * (n - 2) //2)
            assert(len(sampler.res['PARAMS_PROPS'][i]['TREE_ID']) == n * (n - 1) // 2)
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
                accepted_indexes = np.where(sampler.res['ACCEPT_INDEX'][:i + 1])[0]
                if len(accepted_indexes) > 0:
                    last_accepted_idx = np.max(accepted_indexes)
                else:
                    continue
                b_str1 = sampler.res['PARAMS_PROPS'][last_accepted_idx]['BASIS_ID']
                b_str2 = sampler.res['PARAMS_PROPS'][i]['BASIS_ID']
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
        assert(sampler_types_are_correct(n, sampler))
        assert(if_not_similar_then_median_has_larger_lik(g, median_graph, sampler))

def tree_changes_only_every_k_iter(sampler, k):
    trees = [pp['TREE_ID'] for pp in sampler.res['PARAMS_PROPS']]
    tree_change = (np.where(list(map(lambda t, t_: t != t_, trees[:-1], trees[1:])))[0] + 1)
    try:
        for i in tree_change:
            assert(i % k == 0 or (i - 1) %k == 0)
    except AssertionError:
        return False
    return True

def if_both_empty_or_complete_cob_always_happen(sampler, k):
    t = len(sampler.res['SAMPLES'])
    trees = [pp['TREE_ID'] for pp in sampler.res['PARAMS_PROPS']]
    empty = '0' * len(sampler.res['SAMPLES'][0])
    complete = '1' * len(sampler.res['SAMPLES'][0])
    ct = k - 1
    while ct < t - 1:
        if (sampler.res['SAMPLES'][ct] == empty) or (sampler.res['SAMPLES'][ct] == complete):
            try:
                assert(sampler.res['ALPHAS'][ct + 1] == 0)
                assert(sampler.res['ACCEPT_INDEX'][ct + 1])
                assert(trees[ct] != trees[ct + 1])
            except AssertionError:
                return False
            ct += k
    return True

def test_proposal_COB_empty():
    n, n_obs = (5, 500)
    tree_prior = dists.TreePriors.Hubs(n)
    ct_prior = dists.CountPriors.TruncatedNB(1, .5)
    cob_freq = 10

    prior = dists.Priors.BasisCount(n, GraphAndBasis, ct_prior, tree_prior)
    temp = Graph(5)
    g = GraphAndBasis(5, temp._dol, tree=tree_prior.Sample(), basis=np.ones((n-1) * (n-2) // 2, dtype=bool))
    data = generate_data(n, n_obs, g, seed=123)
    lik = dists.Likelihoods.Delta(g.copy())
    prop = dists.Proposals.BasisWalk(n, GraphAndBasis, tree_prior, skip=cob_freq)

    np.random.seed(123)
    sampler = MCMC_Sampler(prior, prop, lik, data)
    sampler.run(5000, g)

    assert(tree_changes_only_every_k_iter(sampler, cob_freq))
    assert(if_both_empty_or_complete_cob_always_happen(sampler, cob_freq))
    assert(only_propose_one_basis_difference(sampler, cob_freq))
    assert(lookup_contains_all_params(sampler))
    assert(sampler_types_are_correct(n, sampler))

def test_proposal_COB_complete():
    n, n_obs = (5, 500)
    tree_prior = dists.TreePriors.Hubs(n)
    ct_prior = dists.CountPriors.TruncatedNB(1, .5)
    cob_freq = 10

    prior = dists.Priors.BasisCount(n, GraphAndBasis, ct_prior, tree_prior)
    temp = Graph(5)
    temp.SetComplete()
    g = GraphAndBasis(5, temp._dol, tree=tree_prior.Sample(), basis=np.ones((n-1) * (n-2) // 2, dtype=bool))
    data = generate_data(n, n_obs, g, seed=123)
    lik = dists.Likelihoods.Delta(g.copy())
    prop = dists.Proposals.BasisWalk(n, GraphAndBasis, tree_prior, skip=cob_freq)

    np.random.seed(123)
    sampler = MCMC_Sampler(prior, prop, lik, data)
    sampler.run(5000, g)

    assert(tree_changes_only_every_k_iter(sampler, cob_freq))
    assert(if_both_empty_or_complete_cob_always_happen(sampler, cob_freq))
    assert(only_propose_one_basis_difference(sampler, cob_freq))
    assert(lookup_contains_all_params(sampler))
    assert(sampler_types_are_correct(n, sampler))

