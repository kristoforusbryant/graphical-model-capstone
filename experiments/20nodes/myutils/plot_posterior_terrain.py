from utils.Graph import Graph
from utils.laplace_approximation import laplace_approx
from dists.Params import GraphAndBasis as Param
from .configs_comparisons import config_to_path
import warnings
import numpy as np
from scipy.stats import nbinom
import pickle
import matplotlib.pyplot as plt

def get_log_posteriors(config):
    n = config['n']
    n_obs = config['n_obs']
    name = config['true_graph']

    with open(f"data/graph_{config['true_graph']}_{config['n']}_{config['n_obs']}.pkl", 'rb') as handle:
        g = pickle.load (handle)
        g = Param(n, g._dol)

    if n <=5:
        g_list = Graph.get_all_graphs(n)

        delta, D = 3, np.eye(n)
        rv = nbinom(1, .5)

        idx = g.GetID()
        data = np.loadtxt(f"data/{name}_{n}_{n_obs}.dat", delimiter=',')

        U = data.transpose() @ data
        assert(U.shape == D.shape)
        liks = list(map(lambda x: laplace_approx(x._dol, delta + n_obs, D + U) - laplace_approx(x._dol, delta, D), g_list))
        diffs = list(map(lambda x: len(set(g.GetEdgeL()).symmetric_difference(x.GetEdgeL())), g_list))
        priors = list(map(lambda x: np.log(rv.pmf(x)), diffs))
        posteriors = np.array(liks) + np.array(priors)

        return np.array(g._basis_active, dtype=int), [(np.array(list(g_list[i].GetID()), dtype=int), posteriors[i]) for i in range(len(g_list))]

    else:
        with open(config_to_path(config), 'rb') as handle:
            sampler = pickle.load(handle)

        if config['cob_freq'] is not None:
            warnings.warn("Bases change throughout the simulation, all graphs is represented w.r.t. the last basis")

        tree = sampler.last_params._tree
        g = Param(len(g), g._dol, tree)
        gba = np.array(g._basis_active, dtype=int)

        log_posts = []
        ids = []
        for i in range(sampler.iter):
            if sampler.res['PARAMS'][i] not in ids:
                g_ba = np.array(list(sampler.res['PARAMS_PROPS'][i]['BASIS_ID']), dtype=int)
                log_posts.append((g_ba, sampler.res['LIK_'][i] + sampler.res['PRIOR'][i]))
                ids.append(sampler.res['PARAMS'][i])
        return gba, log_posts


def plot_posterior_terrain(config):
    gba, lp = get_log_posteriors(config)

    diffs = np.array([((gba + g_ba) % 2).sum() for g_ba, _ in  lp])
    posts = np.array([v for _, v in lp])

    is_superset = np.array([(set(np.where(gba)[0]) - set(np.where(g_ba)[0])) == set() for g_ba,_ in lp], dtype=bool)

    fig = plt.figure(figsize=(10,10))

    #plt.scatter(np.abs(diffs), posts)
    plt.scatter(np.abs(diffs)[np.logical_not(is_superset)], posts[np.logical_not(is_superset)], label='not superset of g', s=15)
    plt.scatter(np.abs(diffs)[is_superset], posts[is_superset], label='superset of g', s=15)
    plt.xlabel("n basis different from true graph", fontsize=20)
    plt.ylabel("log posterior", fontsize=20)

    plt.legend()
    plt.show()
    return gba, diffs, posts, is_superset

