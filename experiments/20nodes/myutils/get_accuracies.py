from utils.diagnostics import str_list_to_adjm, IAC_time
from utils.config_utils import config_to_path
import pickle
import numpy as np

def get_accuracies(config):
    with open(config_to_path(config), 'rb') as handle:
        sampler = pickle.load(handle)
    with open(f"data/graph_{config['true_graph']}_{config['n']}_{config['n_obs']}.pkl", 'rb') as handle:
        g = pickle.load (handle)

    adjm = str_list_to_adjm(len(g), sampler.res['SAMPLES'])
    median_g = (adjm > .5).astype(int)

    def _get_accuracies(sampler, g, md):
        l1= np.array(g.GetBinaryL(), dtype=bool)
        triu = np.triu_indices(len(g), 1)
        l2 = np.array(md[triu], dtype=bool)

        TP = np.logical_and(l1, l2).astype(int).sum()
        TN = np.logical_and(np.logical_not(l1), np.logical_not(l2)).astype(int).sum()
        FP = np.logical_and(np.logical_not(l1), l2).astype(int).sum()
        FN = np.logical_and(l1, np.logical_not(l2)).astype(int).sum()

        assert(TP + TN + FP + FN == len(l1))
        assert(TP + FP == l2.astype(int).sum())
        assert(TN + FN == np.logical_not(l2).astype(int).sum())

        return TP, TN, FP, FN

    return _get_accuracies(sampler, g, median_g)


def get_summary(config, b=0):
    with open(config_to_path(config), 'rb') as handle:
        sampler = pickle.load(handle)

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

    posts = np.array(sampler.res['LIK'], dtype=float)[b:] + np.array(sampler.res['PRIOR'], dtype=float)[b:]
    sizes = list(map(lambda s: np.sum(_str_to_int_list(s)), sampler.res['SAMPLES']))[b:]
    n_bases = _get_basis_ct(sampler)[b:]
    trees = [pp['TREE_ID'] for pp in sampler.res['PARAMS_PROPS']]
    change_tree = np.where(list(map(lambda t, t_: t != t_, trees[:-1], trees[1:])))[0] + 1

    d = {}
    d['IAT_posterior'] = IAC_time(posts)
    d['IAT_sizes'] = IAC_time(sizes)
    d['IAT_bases'] = IAC_time(n_bases)

    d['accept_rate'] = np.sum(sampler.res['ACCEPT_INDEX']) / len(sampler.res['ACCEPT_INDEX'])
    d['tree_accept_ct'] = len(set(change_tree).intersection(set(np.where(sampler.res['ACCEPT_INDEX'])[0])))
    d['max_posterior'] = np.max(posts)
    d['states_visited'] = len(np.unique(sampler.res['SAMPLES'][b:]))
    d['time'] = sampler.time

    return d