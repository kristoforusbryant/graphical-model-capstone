from utils.MCMC import MCMC_Sampler
from collections import OrderedDict
from dists.Likelihoods import GW_LA as Likelihood
import pickle
import numpy as np


DEFAULT_CONFIG =  OrderedDict({ 'n': 5,
                    'n_obs': 250,
                    'init': 'fixed',
                    'true_graph': 'empty',
                    'prior': 'basis-inclusion',
                    'basis': 'edge',
                    'proposal': 'naive',
                    'cob_freq': 0,
                    'iter': int(1e4),
                    'seed': 123 })

ALL_CONFIGS = OrderedDict({ 'n': [5],
                'n_obs': [250],
                'init': ['fixed', 'empty', 'complete', 'circle'],
                'true_graph': 'empty',
                'prior': ['basis-inclusion', 'basis-count'],
                'basis': ['edge', 'hub', 'uniform'],
                'proposal': ['naive', 'BD'],
                'cob_freq': [None, 10, 100],
                'iter': [int(1e4)],
                'seed': 123 })

from dists.TreePriors import Uniform, Hubs

def config_to_path(config):
    path = "results/sampler"
    for k in config.keys():
        path += f"_{k}-{config[k]}"
    return path + ".pkl"

def get_sampler(config):
    with open(config_to_path(config), 'rb') as handle:
        sampler = pickle.load(handle)
    return sampler

def get_config_l(config):
    from itertools import product
    lol = []
    for _, v in config.items():
        if isinstance(v, list):
            lol.append(v)
        else:
            lol.append([v])
    p = product(*lol)
    keys = list(config.keys())
    lod = [{keys[i]: l[i] for i in range(len(l))} for l in p]
    return lod


def parse_init(conf):
    n = conf['n']
    n_obs = conf['n_obs']
    if conf['init'] == 'fixed':
        g = conf['true_graph']
        with open(f'data/graph_{g}_{n}_{n_obs}.pkl', 'rb') as handle:
            return pickle.load(handle)
    elif conf['init'] == 'empty':
        with open(f'data/graph_empty_{n}_{n_obs}.pkl', 'rb') as handle:
            return pickle.load(handle)
    elif conf['init'] == 'complete':
        with open(f'data/graph_complete_{n}_{n_obs}.pkl', 'rb') as handle:
            return pickle.load(handle)
    elif conf['init'] == 'circle':
        with open(f'data/graph_circle_{n}_{n_obs}.pkl', 'rb') as handle:
            return pickle.load(handle)
    elif conf['init'] == 'ml_forest':
        g = conf['true_graph']
        data = np.loadtxt(f"data/{g}_{n}_{n_obs}.dat", delimiter=',')
        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)

        from utils.ML_spanning_tree import ML_forest
        T = ML_forest(data)
        if conf['basis'] == 'edge':
            return T
        elif conf['basis'] == 'uniform' or conf['basis'] == 'hub' or conf['basis'] == 'path':
            return T.GetClosestInCycleSpace()
        else:
            raise ValueError(f"Unrecognized value of conf['basis']: {conf['basis']}")
    elif conf['init'] == 'conv_edges':
        path_pref = f"results/sampler_n-{conf['n']}_n_obs-{conf['n_obs']}_init-empty_true_graph-{conf['true_graph']}_prior-{conf['prior']}_basis-edge"
        import os
        path = [s for s in os.listdir('results') if (path_pref in s) and ('iter-30000' in s) and ('.short' in s)][0]
        with open(path, 'rb') as handle:
            sampler = pickle.load(handle)
        return sampler.last_params
    else:
        raise ValueError(f"Unrecognized value of conf['init']: {conf['init']}")

def parse_prior(conf, Param):
    from dists.CountPriors import TruncatedNB
    from dists.Priors import BasisInclusion, BasisCount

    n = conf['n']
    if conf['prior'] == 'basis-inclusion':
        if conf['basis'] == 'edge':
            return BasisInclusion(n, Param, alpha=.5)
        elif conf['basis'] == 'hub':
            tree_prior = Hubs(n)
            return BasisInclusion(n, Param, alpha=.5, tree_prior=tree_prior)
        elif conf['basis'] == 'uniform':
            tree_prior = Uniform(n)
            return BasisInclusion(n, Param, alpha=.5, tree_prior=tree_prior)
        else:
            raise ValueError(f"Unrecognized value of conf['basis']: {conf['basis']}")

    elif conf['prior'] == 'basis-count':
        ct_prior = TruncatedNB(1, .5)
        if conf['basis'] == 'edge':
            return BasisCount(n, Param, ct_prior)
        elif conf['basis'] == 'hub':
            ct_prior = TruncatedNB(1, .5)
            tree_prior = Hubs(n)
            return BasisCount(n, Param, ct_prior, tree_prior)
        elif conf['basis'] == 'uniform':
            ct_prior = TruncatedNB(1, .5)
            tree_prior = Uniform(n)
            return BasisCount(n, Param, ct_prior, tree_prior)
        else:
            raise ValueError(f"Unrecognized value of conf['basis']: {conf['basis']}")
    else:
        raise ValueError(f"Unrecognized value of conf['prior']: {conf['prior']}")


def parse_proposal(conf, Param):
    from dists.Proposals import BasisWalk, BasisBD
    n = conf['n']
    if conf['proposal'] == 'naive':
        if conf['basis'] == 'edge':
            return BasisWalk(n, Param)
        elif conf['basis'] == 'hub':
            tree_prior = Hubs(n)
            return BasisWalk(n, Param, tree_prior, conf['cob_freq'])
        elif conf['basis'] == 'uniform':
            tree_prior = Uniform(n)
            return BasisWalk(n, Param, tree_prior, conf['cob_freq'])
        else:
            raise ValueError(f"Unrecognized value of conf['basis']: {conf['basis']}")
    elif conf['proposal'] == 'BD':
        if conf['basis'] == 'edge':
            return BasisBD(n, Param)
        elif conf['basis'] == 'hub':
            tree_prior = Hubs(n)
            return BasisBD(n, Param, tree_prior, conf['cob_freq'])
        elif conf['basis'] == 'uniform':
            tree_prior = Uniform(n)
            return BasisBD(n, Param, tree_prior, conf['cob_freq'])
        else:
            raise ValueError(f"Unrecognized value of conf['basis']: {conf['basis']}")
    else:
        raise ValueError(f"Unrecognized value of conf['proposal']: {conf['proposal']}")


def parse_config(conf, Param):
    n = conf['n']
    init = parse_init(conf)
    prior = parse_prior(conf, Param)
    prop = parse_proposal(conf, Param)

    if conf['basis'] == 'edge':
        init = Param(len(init), init._dol)
    elif conf['basis'] == 'hub' or conf['basis'] == 'path':
        init = Param(len(init), init._dol, tree = prior._tree_prior.Sample())
    elif conf['basis'] == 'uniform':
        if conf['init'] == 'ml_forest':
            n_obs = conf['n_obs']
            g = conf['true_graph']
            data = np.loadtxt(f"data/{g}_{n}_{n_obs}.dat", delimiter=',')
            if len(data.shape) == 1:
                data = data.reshape(data.shape[0], 1)

            from utils.ML_spanning_tree import ML_spanning_tree
            T, _ = ML_spanning_tree(data)
            init = Param(len(init), init._dol, tree = T)
        else:
            # n_obs = conf['n_obs']
            # g = conf['true_graph']
            # data = np.loadtxt(f"data/{g}_{n}_{n_obs}.dat", delimiter=',')
            # if len(data.shape) == 1:
            #     data = data.reshape(data.shape[0], 1)

            # from utils.ML_spanning_tree import ML_spanning_tree
            # T, _ = ML_spanning_tree(data)
            # init = Param(len(init), init._dol, tree = T)
            init = Param(len(init), init._dol, tree = prior._tree_prior.Sample())

    else:
        raise ValueError(f"Unrecognized value of conf['basis']: {conf['basis']}")

    iter = conf['iter']
    seed = conf.get('seed', None)

    return n, init, prior, prop, iter, seed

def run_config(data, conf):
    from dists.Params import GraphAndBasis as Param

    # Parsing
    n, init, prior, prop, iter, seed = parse_config(conf, Param)
    lik = Likelihood(data, 3, np.eye(n), Param, alpha=conf['temper'])

    sampler = MCMC_Sampler(prior, prop, lik, data, outfile=config_to_path(conf))

    if seed:
        np.random.seed(seed)
    sampler.run(iter, init)
    sampler.save_object()

    return sampler