from utils.config_utils import get_config_l, config_to_path
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BETTER_NAMES = {'edge': 'edge',
                'hub': 'star',
                'uniform': 'all',
                'path': 'path',
                'empty': 'empty',
                'circle': 'circle',
                'random0': 'cb_size_based0',
                'random1': 'cb_size_based1',
                'random2': 'erdos-renyi',
                'random3': 'scale-free',
                'random4': 'large0',
                'random5': 'large1',
                'small0': 'small0',
                'small1': 'small1'
                }

PERC_TO_IDX = {.25: 0, .5: 1, .75: 3}

def get_summary(c, b=0, thin=1):
    path = f"{config_to_path(c)[:-4]}_burnin-{b}_thin-{thin}.short"
    with open(path, 'rb') as handle:
        summ = pickle.load(handle)
    return summ.summary

def plot_distances(configs, basis_list=['edge', 'hub', 'uniform'], burnin=0, thin=1, uniq=False, proposed=False, plot=False):
    n, n_obs = configs['n'],  configs['n_obs']
    fig, axs = plt.subplots(len(configs['true_graph']), 3, figsize=(3 * 10, len(configs['true_graph']) * 10))
    plt.rc('xtick',labelsize=30)
    plt.rc('ytick',labelsize=30)

    # Setting (shared) x and y labels
    names = [BETTER_NAMES[s] for s in configs['true_graph']]

    for i in range(len(configs['true_graph'])):
        axs[i, 0].set_ylabel(names[i], size=50)

    axs[0, 0].set_title('jaccard', size=50)
    axs[0, 1].set_title('hamming', size=50)
    axs[0, 2].set_title('sizes', size=50)

    # Getting Ranges
    jacc_max = [.0] * len(configs['true_graph'])
    hamm_max = [.0] * len(configs['true_graph'])
    size_max = [.0] * len(configs['true_graph'])
    for basis in basis_list:
        configs['basis'] = basis
        config_l = get_config_l(configs)
        summaries = [ get_summary(c, burnin, thin) for c in config_l ]

        for i in range(len(summaries)):
            if not uniq and not proposed:
                if np.max(summaries[i]['jaccard_distances']) * 100 > jacc_max[i]:
                    jacc_max[i] = np.max(summaries[i]['jaccard_distances']) * 100
                if np.max(summaries[i]['hamming_distances']) > hamm_max[i]:
                    hamm_max[i] = np.max(summaries[i]['hamming_distances'])
                if np.max(summaries[i]['size_distances']) > size_max[i]:
                    size_max[i] = np.max(summaries[i]['size_distances'])
            elif uniq and not proposed:
                if np.max(summaries[i]['jaccard_distances_uniq']) * 100 > jacc_max[i]:
                    jacc_max[i] = np.max(summaries[i]['jaccard_distances_uniq']) * 100
                if np.max(summaries[i]['hamming_distances_uniq']) > hamm_max[i]:
                    hamm_max[i] = np.max(summaries[i]['hamming_distances_uniq'])
                if np.max(summaries[i]['size_distances_uniq']) > size_max[i]:
                    size_max[i] = np.max(summaries[i]['size_distances_uniq'])
            elif proposed and not uniq:
                if np.max(summaries[i]['jaccard_distances_']) * 100 > jacc_max[i]:
                    jacc_max[i] = np.max(summaries[i]['jaccard_distances_']) * 100
                if np.max(summaries[i]['hamming_distances_']) > hamm_max[i]:
                    hamm_max[i] = np.max(summaries[i]['hamming_distances_'])
                if np.max(summaries[i]['size_distances_']) > size_max[i]:
                    size_max[i] = np.max(summaries[i]['size_distances_'])
            else:
                if np.max(summaries[i]['jaccard_distances_uniq_']) * 100 > jacc_max[i]:
                    jacc_max[i] = np.max(summaries[i]['jaccard_distances_uniq_']) * 100
                if np.max(summaries[i]['hamming_distances_uniq_']) > hamm_max[i]:
                    hamm_max[i] = np.max(summaries[i]['hamming_distances_uniq_'])
                if np.max(summaries[i]['size_distances_uniq_']) > size_max[i]:
                    size_max[i] = np.max(summaries[i]['size_distances_uniq_'])

    # Plotting
    for basis in basis_list:
        configs['basis'] = basis
        config_l = get_config_l(configs)
        summaries = [ get_summary(c, burnin, thin) for c in config_l ]

        for i in range(len(summaries)):
            if not uniq and not proposed:
                axs[i, 0].hist(summaries[i]['jaccard_distances'], bins=np.arange(jacc_max[i] + 1) / 100, label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 1].hist(summaries[i]['hamming_distances'], bins=np.arange(hamm_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 2].hist(summaries[i]['size_distances'], bins=np.arange(size_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)
            elif uniq and not proposed:
                axs[i, 0].hist(summaries[i]['jaccard_distances_uniq'], bins=np.arange(jacc_max[i] + 1) / 100, label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 1].hist(summaries[i]['hamming_distances_uniq'], bins=np.arange(hamm_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 2].hist(summaries[i]['size_distances_uniq'], bins=np.arange(size_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)
            elif proposed and not uniq:
                axs[i, 0].hist(summaries[i]['jaccard_distances_'], bins=np.arange(jacc_max[i] + 1) / 100, label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 1].hist(summaries[i]['hamming_distances_'], bins=np.arange(hamm_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 2].hist(summaries[i]['size_distances_'], bins=np.arange(size_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)
            else:
                axs[i, 0].hist(summaries[i]['jaccard_distances_uniq_'], bins=np.arange(jacc_max[i] + 1) / 100, label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 1].hist(summaries[i]['hamming_distances_uniq_'], bins=np.arange(hamm_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)
                axs[i, 2].hist(summaries[i]['size_distances_uniq_'], bins=np.arange(size_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5)

            axs[i, 0].legend(fontsize=30)
            axs[i, 1].legend(fontsize=30)
            axs[i, 2].legend(fontsize=30)

    if not uniq and not proposed:
        fig.savefig(f"distances_distr_n-{n}_n_obs-{n_obs}.pdf")
    elif uniq and not proposed:
        fig.savefig(f"distances_u_distr_n-{n}_n_obs-{n_obs}.pdf")
    elif proposed and not uniq:
        fig.savefig(f"distances_p_distr_n-{n}_n_obs-{n_obs}.pdf")
    else:
        fig.savefig(f"distances_u_p_distr_n-{n}_n_obs-{n_obs}.pdf")

    if plot:
        plt.show()

    return fig

# Univariate variances statistics from (Scutari 2013) https://doi.org/10.1214/13-BA819

def get_vars(cl, b=0, thin=1):
    summ = [ get_summary(c, b, thin) for c in cl ]
    variances = [(round(x['tvar'], 3),  round(x['tvar_'], 3)) for x in summ]
    return variances

def compare_variances(configs, basis_list=['edge', 'hub', 'uniform'], burnin=0, thin=1):
    n, n_obs = configs['n'],  configs['n_obs']

    nrows = len(configs['true_graph'])
    ncols = len(basis_list) * 2
    data = np.zeros((nrows, ncols))

    for i in range(len(basis_list)):
        configs['basis'] = basis_list[i]
        config_l = get_config_l(configs)
        vars = get_vars(config_l, burnin, thin)
        data[:, i] = [ tup[0] for tup in vars ]
        data[:, len(basis_list) + i] = [ tup[1] for tup in vars ]

    basis_names = [BETTER_NAMES[s] for s in basis_list]
    columns = pd.MultiIndex.from_product([['tvar', 'tvar_p'], basis_names])
    indexes = [BETTER_NAMES[s] for s in configs['true_graph']]
    df = pd.DataFrame(data, index=indexes, columns=columns)

    return df

# Accuracies
def get_accuracies(c, b=0, thin=1):
    path = f"{config_to_path(c)[:-4]}_burnin-{b}_thin-{thin}.short"
    with open(path, 'rb') as handle:
        summ = pickle.load(handle)
    return summ.accuracies

def compare_accuracies(configs, basis_list=['edge', 'hub', 'uniform'], burnin=0, thin=1, percentile=.5):
    n, n_obs = configs['n'],  configs['n_obs']

    nrows = len(configs['true_graph'])
    ncols = len(basis_list) * 2
    data = np.zeros((nrows, ncols))

    i = PERC_TO_IDX[percentile]
    for i in range(len(basis_list)):
        configs['basis'] = basis_list[i]
        config_l = get_config_l(configs)
        acc_l = [ get_accuracies(c, burnin, thin) for c in config_l ]
        data[:, i] = [round(x[i][2], 3) for x in acc_l]
        data[:, len(basis_list) + i] = [ round(x[i][3], 3) for x in acc_l]

    basis_names = [BETTER_NAMES[s] for s in basis_list]
    columns = pd.MultiIndex.from_product([['FP', 'FN'], basis_names])
    indexes = [BETTER_NAMES[s] for s in configs['true_graph']]
    df = pd.DataFrame(data, index=indexes, columns=columns)

    return df

# Acceptance Rates
def compare_acceptance(configs, basis_list=['edge', 'hub', 'uniform'], burnin=0, thin=1):
    n, n_obs = configs['n'],  configs['n_obs']

    nrows = len(configs['true_graph'])
    ncols = len(basis_list) * 2
    data = np.zeros((nrows, ncols))

    for i in range(len(basis_list)):
        configs['basis'] = basis_list[i]
        config_l = get_config_l(configs)
        summ = [ get_summary(c, burnin, thin) for c in config_l ]
        data[:, i] = [round(x['accept_rate'], 3) for x in summ]
        data[:, len(basis_list) + i] = [ round(x['tree_accept_ct']  / (configs['iter'] / configs['cob_freq']), 3) for x in summ]

    basis_names = [BETTER_NAMES[s] for s in basis_list]
    columns = pd.MultiIndex.from_product([['accept', 'accept_tree'], basis_names])
    indexes = [BETTER_NAMES[s] for s in configs['true_graph']]
    df = pd.DataFrame(data, index=indexes, columns=columns)

    return df

# Acceptance Scaled distances

def plot_start(configs, basis_list=['edge', 'hub', 'uniform'], burnin=0, thin=1, plot=False):
    n, n_obs = configs['n'],  configs['n_obs']
    fig, axs = plt.subplots(len(configs['true_graph']), 3, figsize=(3 * 10, len(configs['true_graph']) * 10))
    plt.rc('xtick',labelsize=30)
    plt.rc('ytick',labelsize=30)

    # Setting (shared) x and y labels
    names = [BETTER_NAMES[s] for s in configs['true_graph']]

    for i in range(len(configs['true_graph'])):
        axs[i, 0].set_ylabel(names[i], size=50)

    axs[0, 0].set_title('jaccard', size=50)
    axs[0, 1].set_title('hamming', size=50)
    axs[0, 2].set_title('sizes', size=50)

    # Getting Ranges
    jacc_max = [.0] * len(configs['true_graph'])
    hamm_max = [.0] * len(configs['true_graph'])
    size_max = [.0] * len(configs['true_graph'])
    for basis in basis_list:
        configs['basis'] = basis
        config_l = get_config_l(configs)
        summaries = [ get_summary(c, burnin, thin) for c in config_l ]

        for i in range(len(summaries)):
            if len(summaries[i]['jaccard_distances_start']) == 0:
                print(config_to_path(config_l[i]))
                summaries[i]['jaccard_distances_start'] = [0]
            if len(summaries[i]['hamming_distances_start']) == 0:
                print(config_to_path(config_l[i]))
                summaries[i]['hamming_distances_start'] = [0]
            if len(summaries[i]['size_distances_start']) == 0:
                print(config_to_path(config_l[i]))
                summaries[i]['size_distances_start'] = [0]

            if np.max(summaries[i]['jaccard_distances_start']) * 100 > jacc_max[i]:
                jacc_max[i] = np.max(summaries[i]['jaccard_distances_start']) * 100
            if np.max(summaries[i]['hamming_distances_start']) > hamm_max[i]:
                hamm_max[i] = np.max(summaries[i]['hamming_distances_start'])
            if np.max(summaries[i]['size_distances_start']) > size_max[i]:
                size_max[i] = np.max(summaries[i]['size_distances_start'])

    # Plotting
    for basis in basis_list:
        configs['basis'] = basis
        config_l = get_config_l(configs)
        summaries = [ get_summary(c, burnin, thin) for c in config_l ]

        for i in range(len(summaries)):
            axs[i, 0].hist(summaries[i]['jaccard_distances_start'], bins=np.arange(jacc_max[i] + 1) / 100, label=BETTER_NAMES[basis], alpha=.5, density=True)
            axs[i, 1].hist(summaries[i]['hamming_distances_start'], bins=np.arange(hamm_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5, density=True)
            axs[i, 2].hist(summaries[i]['size_distances_start'], bins=np.arange(size_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5, density=True)

            axs[i, 0].legend(fontsize=30)
            axs[i, 1].legend(fontsize=30)
            axs[i, 2].legend(fontsize=30)

    fig.savefig(f"as_start_distr_n-{n}_n_obs-{n_obs}.pdf")

    if plot:
        plt.show()

    return fig

def plot_end(configs, basis_list=['edge', 'hub', 'uniform'], burnin=0, thin=1, plot=False):
    n, n_obs = configs['n'],  configs['n_obs']
    fig, axs = plt.subplots(len(configs['true_graph']), 3, figsize=(3 * 10, len(configs['true_graph']) * 10))
    plt.rc('xtick',labelsize=30)
    plt.rc('ytick',labelsize=30)

    # Setting (shared) x and y labels
    names = [BETTER_NAMES[s] for s in configs['true_graph']]

    for i in range(len(configs['true_graph'])):
        axs[i, 0].set_ylabel(names[i], size=50)

    axs[0, 0].set_title('jaccard', size=50)
    axs[0, 1].set_title('hamming', size=50)
    axs[0, 2].set_title('sizes', size=50)

    # Getting Ranges
    jacc_max = [.0] * len(configs['true_graph'])
    hamm_max = [.0] * len(configs['true_graph'])
    size_max = [.0] * len(configs['true_graph'])
    for basis in basis_list:
        configs['basis'] = basis
        config_l = get_config_l(configs)
        summaries = [ get_summary(c, burnin, thin) for c in config_l ]

        for i in range(len(summaries)):
            if len(summaries[i]['jaccard_distances_end']) == 0:
                print(config_to_path(config_l[i]))
                summaries[i]['jaccard_distances_end'] = [0]
            if len(summaries[i]['hamming_distances_end']) == 0:
                print(config_to_path(config_l[i]))
                summaries[i]['hamming_distances_end'] = [0]
            if len(summaries[i]['size_distances_end']) == 0:
                print(config_to_path(config_l[i]))
                summaries[i]['size_distances_end'] = [0]

            if np.max(summaries[i]['jaccard_distances_end']) * 100 > jacc_max[i]:
                jacc_max[i] = np.max(summaries[i]['jaccard_distances_end']) * 100
            if np.max(summaries[i]['hamming_distances_end']) > hamm_max[i]:
                hamm_max[i] = np.max(summaries[i]['hamming_distances_end'])
            if np.max(summaries[i]['size_distances_end']) > size_max[i]:
                size_max[i] = np.max(summaries[i]['size_distances_end'])

    # Plotting
    for basis in basis_list:
        configs['basis'] = basis
        config_l = get_config_l(configs)
        summaries = [ get_summary(c, burnin, thin) for c in config_l ]

        for i in range(len(summaries)):
            axs[i, 0].hist(summaries[i]['jaccard_distances_end'], bins=np.arange(jacc_max[i] + 1) / 100, label=BETTER_NAMES[basis], alpha=.5, density=True)
            axs[i, 1].hist(summaries[i]['hamming_distances_end'], bins=np.arange(hamm_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5, density=True)
            axs[i, 2].hist(summaries[i]['size_distances_end'], bins=np.arange(size_max[i] + 1), label=BETTER_NAMES[basis], alpha=.5, density=True)

            axs[i, 0].legend(fontsize=30)
            axs[i, 1].legend(fontsize=30)
            axs[i, 2].legend(fontsize=30)

    fig.savefig(f"as_end_distr_n-{n}_n_obs-{n_obs}.pdf")

    if plot:
        plt.show()

    return fig


def get_as_vars(cl, b=0, thin=1):
    summ = [ get_summary(c, b, thin) for c in cl ]
    variances = [(round(x['as_start_tvar'], 3),  round(x['as_end_tvar'], 3)) for x in summ]
    return variances

def compare_as_variances(configs, basis_list=['edge', 'hub', 'uniform'], burnin=0, thin=1):
    n, n_obs = configs['n'],  configs['n_obs']

    nrows = len(configs['true_graph'])
    ncols = len(basis_list) * 2
    data = np.zeros((nrows, ncols))

    for i in range(len(basis_list)):
        configs['basis'] = basis_list[i]
        config_l = get_config_l(configs)
        vars = get_as_vars(config_l, burnin, thin)
        data[:, i] = [ tup[0] for tup in vars ]
        data[:, len(basis_list) + i] = [ tup[1] for tup in vars ]

    basis_names = [BETTER_NAMES[s] for s in basis_list]
    columns = pd.MultiIndex.from_product([['as_start_tvar', 'as_end_tvar'], basis_names])
    indexes = [BETTER_NAMES[s] for s in configs['true_graph']]
    df = pd.DataFrame(data, index=indexes, columns=columns)

    return df
