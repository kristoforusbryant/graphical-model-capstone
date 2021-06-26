import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils.config_utils import config_to_path, get_config_l
from .plot_posterior_dict import plot_true_posterior, plot_true_posterior_cb_marginalized, plot_true_posterior_edge_marginalized, get_post_dict_cb_only

def MC_to_post_dict(n, name):
    a = np.loadtxt(f"MC_true_post/MC_{name}.dat")
    post_dict = {np.binary_repr(i, width=(n * (n - 1) // 2)): a[i] for i in range(len(a))}
    return post_dict

def _str_to_int_list(s):
    return np.array(list(s), dtype=int)

def _get_basis_ct(sampler):
    basis_ct = []
    if sampler.res['ACCEPT_INDEX'][0] == 0:
        basis_ct.append(np.sum(_str_to_int_list(sampler.init['BASIS_ID'])))
    else:
        basis_ct.append(np.sum(_str_to_int_list(sampler.res['PARAMS_PROPS'][0]['BASIS_ID'])))

    for i in range(1, len(sampler.res['ACCEPT_INDEX'])):
        if sampler.res['ACCEPT_INDEX'][i]:
            basis_ct.append(np.sum(_str_to_int_list(sampler.res['PARAMS_PROPS'][i]['BASIS_ID'])))
        else:
            basis_ct.append(basis_ct[-1])
    return basis_ct

def _get_varying(configs):
    l = [ (k, v) for k, v in configs.items() if (isinstance(v, list)) and (len(v) > 1)]
    assert len(l) == 1, "more than one varying config variable"
    return l[0]

def sampler_to_post_dict(sampler, all_graphs):
    from collections import Counter
    post_dict = Counter(sampler.res['SAMPLES'])
    iter = len(sampler.res['SAMPLES'])
    for s in all_graphs:
        if s in list(post_dict.keys()):
            post_dict[s] = post_dict[s] / iter
        else:
            post_dict[s] = 0
    return post_dict

def compare_traces(configs, log=False, burnin=0):
    config_l = get_config_l(configs)
    varying_k, varying_v = _get_varying(configs)
    paths = [config_to_path(c) for c in config_l]
    cols = len(paths)

    all_visited_states = set()
    for c in config_l:
        with open(config_to_path(c), 'rb') as handle:
            sampler = pickle.load(handle)
        all_visited_states = all_visited_states.union(set(np.unique(sampler.res['SAMPLES'])))

    posts = []
    post_traces = []
    size_traces = []
    basis_traces = []
    init_bases = []
    for c in config_l:
        with open(config_to_path(c), 'rb') as handle:
            sampler = pickle.load(handle)
        post = sampler_to_post_dict(sampler, list(all_visited_states))
        if c['basis'] != 'edge':
            post = get_post_dict_cb_only(c['n'], post)
        posts.append(post)
        post_traces.append(np.array(sampler.res['LIK']) + np.array(sampler.res['PRIOR']))
        size_traces.append(list(map(lambda s: np.sum(_str_to_int_list(s)), sampler.res['SAMPLES'])))
        basis_traces.append(_get_basis_ct(sampler))
        init_bases.append(sampler.last_params._basis)


    fig, axs = plt.subplots(3, cols + 1, figsize=(10 * (cols + 1), 10 * 3))

    for i in range(cols):
        plot_true_posterior(posts[i], log, ax=axs[0, 0], label=f"{varying_k}: {varying_v[i]}")
        plot_true_posterior_edge_marginalized(posts[i], log, ax=axs[1, 0], label=f"{varying_k}: {varying_v[i]}")

        if config_l[i]['cob_freq'] is None and config_l[i]['basis'] != 'edge':
            basis = init_bases[i]
            with open(config_to_path(c), 'rb') as handle:
                sampler = pickle.load(handle)
            plot_true_posterior_cb_marginalized(posts[i], basis, log, ax=axs[2, 0], sampler=sampler)


        axs[0, i + 1].plot(post_traces[i][burnin:])
        axs[1, i + 1].plot(size_traces[i][burnin:])
        axs[2, i + 1].plot(basis_traces[i][burnin:])

    axs[0, 0].legend()
    axs[1, 0].legend()

    for i in range(cols):
        axs[0, i + 1].set_title(f"{varying_k}: {varying_v[i]}", fontsize=20)

    ylabs = ["MCMC posterior", "sizes", "n_basis"]
    for i in range(len(ylabs)):
        axs[i, 0].set_ylabel(ylabs[i], rotation= 90, fontsize=20)

    plt.show()
    return posts

def compare_traces_short(configs, log=False, burnin=0):
    config_l = get_config_l(configs)
    varying_k, varying_v = _get_varying(configs)
    paths = [config_to_path(c) for c in config_l]
    cols = len(paths)

    post_traces = []
    size_traces = []
    basis_traces = []
    init_bases = []
    for c in config_l:
        with open(config_to_path(c)[-4] + f"_burnin-0.short", 'rb') as handle:
            sampler = pickle.load(handle)
        post_traces.append(sampler.posteriors)
        size_traces.append(sampler.sizes)
        basis_traces.append(sampler.bases)
        init_bases.append(sampler.last_params._basis)

    fig, axs = plt.subplots(3, cols, figsize=(10 * (cols), 10 * 3))

    for i in range(cols):
        axs[0, i].plot(post_traces[i][burnin:])
        axs[1, i].plot(size_traces[i][burnin:])
        axs[2, i].plot(basis_traces[i][burnin:])

    for i in range(cols):
        axs[0, i].set_title(f"{varying_k}: {varying_v[i]}", fontsize=20)

    ylabs = ["MCMC posterior", "sizes", "n_basis"]
    for i in range(len(ylabs)):
        axs[i, 0].set_ylabel(ylabs[i], rotation= 90, fontsize=20)

    plt.show()


def compare_with_true_posterior(config, burnin=0, log=False):
    n = config['n']
    n_obs = config['n_obs']
    name = config['true_graph']

    MC_post = MC_to_post_dict(n, name)
    with open(f"results/true_posterior_{name}_{n}_{n_obs}.pkl", 'rb') as handle:
        LA_post = pickle.load(handle)
    with open(config_to_path(config), 'rb') as handle:
        sampler = pickle.load(handle)
    MCMC_post = sampler_to_post_dict(sampler, MC_post.keys())

    if config['basis'] != 'edge':
        MC_post = get_post_dict_cb_only(n, MC_post)
        LA_post = get_post_dict_cb_only(n, LA_post)
        MCMC_post = get_post_dict_cb_only(n, MCMC_post)

    if config['cob_freq'] is None and config['basis'] != 'edge':
        fig, axs = plt.subplots(3, 2, figsize=(10 * 2, 10 * 3))
    else:
         fig, axs = plt.subplots(2, 2, figsize=(10 * 2, 10 * 2))

    plot_true_posterior(MC_post, log, ax=axs[0, 0], label="MC")
    plot_true_posterior(LA_post, log, ax=axs[0, 0], label="LA")
    plot_true_posterior(MCMC_post, log, ax=axs[0, 0], label="MCMC")
    axs[0, 0].legend()

    plot_true_posterior_edge_marginalized(MC_post, log, ax=axs[1, 0], label="MC")
    plot_true_posterior_edge_marginalized(LA_post, log, ax=axs[1, 0], label="LA")
    plot_true_posterior_edge_marginalized(MCMC_post, log, ax=axs[1, 0], label="MCMC")
    axs[1, 0].legend()

    if config['cob_freq'] is None and config['basis'] != 'edge':
        basis = sampler.last_params._basis
        plot_true_posterior_cb_marginalized(MC_post, basis, log, ax=axs[2, 0], label="MC")
        plot_true_posterior_cb_marginalized(LA_post, basis, log, ax=axs[2, 0], label="LA")
        plot_true_posterior_cb_marginalized(MCMC_post, basis, log, ax=axs[2, 0], label="MCMC")
        axs[2, 0].legend()


    posterior = np.array(sampler.res['LIK']) + np.array(sampler.res['PRIOR'])
    sizes = list(map(lambda s: np.sum(_str_to_int_list(s)), sampler.res['SAMPLES']))
    n_bases = _get_basis_ct(sampler)

    axs[0, 1].plot(posterior[burnin:])
    axs[1, 1].plot(sizes[burnin:])
    if config['cob_freq'] is None and config['basis'] != 'edge':
        axs[2, 1].plot(n_bases[burnin:])

    axs[0, 0].set_title("compare_w_true", fontsize=20)
    axs[0, 1].set_title(f"traces", fontsize=20)

    ylabs = ["graph_index", "edges"]
    for i in range(len(ylabs)):
        axs[i, 0].set_ylabel(ylabs[i], rotation= 90, fontsize=20)

    ylabs = ["log posterior", "sizes"]
    for i in range(len(ylabs)):
        axs[i, 1].set_ylabel(ylabs[i], rotation= 90, fontsize=20)

    if config['cob_freq'] is None and config['basis'] != 'edge':
        axs[2, 0].set_ylabel("basis", rotation= 90, fontsize=20)
        axs[2, 1].set_ylabel("n_basis", rotation= 90, fontsize=20)

    plt.show()
    return MC_post, LA_post, MCMC_post


from utils.config_utils import get_config_l, config_to_path
import matplotlib.pyplot as plt
import pickle
from utils.diagnostics import str_list_to_median_graph
from utils.Graph import Graph

def _get_varying(configs):
    l = [ (k, v) for k, v in configs.items() if (isinstance(v, list)) and (len(v) > 1)]
    assert(len(l) == 1)
    return l[0]

def compare_median_graphs(configs, threshold=.5, how=None):
    config_l = get_config_l(configs)
    varying_k, varying_v = _get_varying(configs)
    paths = [config_to_path(c) for c in config_l]
    cols = len(paths)

    fig, axs = plt.subplots(1, cols + 1, figsize=(10 * (cols + 1), 10))

    n, n_obs, true_g = config_l[0]['n'], config_l[0]['n_obs'], config_l[0]['true_graph']
    pos = Graph(n).GetCirclePos()

    with open(f"data/graph_{true_g}_{n}_{n_obs}.pkl", 'rb') as handle:
        g = pickle.load (handle)
    if how == 'circle':
        g.Draw(ax=axs[0], pos=pos)
    else:
        g.Draw(ax=axs[0])
    axs[0].set_title('true_graph', fontsize=20)

    for i in range(cols):
        with open(config_to_path(config_l[i]), 'rb') as handle:
            sampler = pickle.load(handle)
        adjm = str_list_to_median_graph(n, sampler.res['SAMPLES'], threshold=threshold)
        g_ = Graph(n)
        g_.SetFromAdjM(adjm)
        if how == 'circle':
            g_.Draw(ax = axs[i + 1], pos=pos)
        else:
            g_.Draw(ax = axs[i + 1])
        axs[i + 1].set_title(f"{varying_k}: {varying_v[i]}", fontsize=20)

    plt.show()
