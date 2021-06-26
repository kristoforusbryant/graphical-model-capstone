from utils.Graph import Graph
from scipy.stats import nbinom
from utils.laplace_approximation import laplace_approx
import numpy as np


def get_edge_order(post_dict):
    """order by number of edges, then by lexicographic order of id"""
    str_to_edge_count = lambda s: np.sum([int(b) for b in list(s)])
    count_str_list = [(str_to_edge_count(s), s) for s in post_dict.keys()]
    return sorted(count_str_list)

def get_cb_order(basis):
    """order by number of basis, then by lexicographic order of id"""
    from itertools import combinations
    count_str_list = [(0, '0'*basis.shape[0])]
    for i in range(1, basis.shape[1] + 1):
        comb = list(combinations(range(basis.shape[1]), i))
        for c in comb:
            bl = np.sum(basis[:, c], axis=1)
            count_str_list.append((i, "".join(np.array(bl, dtype=str))))
    return sorted(count_str_list)

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

def get_cb_order_from_sampler(sampler):
    assert(sampler.prop._skip is None)
    basis_ct = _get_basis_ct(sampler)
    basis_ct_str_list = []
    for i in range(sampler.iter):
        if (basis_ct[i], sampler.res['SAMPLES'][i]) in basis_ct_str_list:
            continue
        basis_ct_str_list.append((basis_ct[i], sampler.res['SAMPLES'][i]))
    return sorted(basis_ct_str_list)

def get_post_dict_cb_only(n, post_dict):
    pd = {}
    for s,v in post_dict.items():
        if not (np.sum(str_to_adj(n, s), axis=0) % 2 == 0).all():
            pd[s] = 0
        else:
            pd[s] = post_dict[s]
    normalising = np.sum(list(pd.values()))
    for s, v in pd.items():
        pd[s] = v / normalising
    return pd

import matplotlib.pyplot as plt
def plot_true_posterior(post_dict, log=False, ax=None, label=""):
    ordered = [post_dict[s] for _,s in get_edge_order(post_dict)]
    if ax is None:
        if not log:
            plt.scatter(range(len(ordered)), ordered, alpha=.5, label=label)
        else:
            plt.scatter(range(len(ordered)), np.log(ordered), alpha=.5, label=label)
    else:
        if not log:
            ax.scatter(range(len(ordered)), ordered, alpha=.5, label=label)
        else:
            ax.scatter(range(len(ordered)), np.log(ordered), alpha=.5, label=label)

def plot_true_posterior_cb_only(n, post_dict, log=False, ax=None, label=""):
    order = get_edge_order(post_dict)
    is_cb = [(np.sum(str_to_adj(n, s), axis=0) % 2 == 0).all() for _,s in order]
    ordered = [post_dict[s] for _,s in order]
    ordered = np.array(ordered) * np.array(is_cb)
    ordered = ordered / np.sum(ordered)
    if ax is None:
        if not log:
            plt.scatter(range(len(ordered)), ordered, alpha=.5, label=label)
        else:
            plt.scatter(range(len(ordered)), np.log(ordered), alpha=.5, label=label)
    else:
        if not log:
            ax.scatter(range(len(ordered)), ordered, alpha=.5, label=label)
        else:
            ax.scatter(range(len(ordered)), np.log(ordered), alpha=.5, label=label)

def plot_true_posterior_edge_marginalized(post_dict, log=False, ax=None, label=""):
    probs = {}
    i_ = 0
    acc = 0
    order = get_edge_order(post_dict)
    for i, s in order:
        if i == i_:
            acc += post_dict[s]
        else:
            probs[i_] = acc
            i_ = i
            acc = post_dict[s]
    probs[i_] = acc
    if ax is None:
        if not log:
            plt.scatter(list(probs.keys()), list(probs.values()), alpha=.5, label=label)
        else:
            plt.scatter(list(probs.keys()), np.log(list(probs.values())), alpha=.5, label=label)
    else:
        if not log:
            ax.scatter(list(probs.keys()), list(probs.values()), alpha=.5, label=label)
        else:
            ax.scatter(list(probs.keys()), np.log(list(probs.values())), alpha=.5, label=label)


def plot_true_posterior_cb_marginalized(post_dict, basis, log=False, ax=None, sampler=None, label=""):
    probs = {}
    i_ = 0
    acc = 0
    if sampler is None:
        order = get_cb_order(basis)
    else:
        order = get_cb_order_from_sampler(sampler)
    for i, s in order:
        if s not in post_dict.keys():
            continue
        if i == i_:
            acc += post_dict[s]
        else:
            probs[i_] = acc
            i_ = i
            acc = post_dict[s]
    probs[i_] = acc
    if ax is None:
        if not log:
            plt.scatter(list(probs.keys()), np.array(list(probs.values())) / np.sum(list(probs.values())), alpha=.5, label=label)
        else:
            plt.scatter(list(probs.keys()), np.log(np.array(list(probs.values())) / np.sum(list(probs.values()))), alpha=.5, label=label)
    else:
        if not log:
            ax.scatter(list(probs.keys()), np.array(list(probs.values())) / np.sum(list(probs.values())), alpha=.5, label=label)
        else:
            ax.scatter(list(probs.keys()), np.log(np.array(list(probs.values())) / np.sum(list(probs.values()))), alpha=.5, label=label)

def str_to_adj(n, s):
    adj = np.zeros((n, n))
    triu = np.triu_indices(n, 1)
    adj[triu] = np.array(list(s), dtype=int)
    return adj + adj.transpose()
