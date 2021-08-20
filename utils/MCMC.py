import numpy as np
from numpy.core.fromnumeric import size
from tqdm import tqdm
import time
import pickle

class MCMC_Sampler:
    def __init__(self, prior, proposal, likelihood, data, outfile=""):
        self.prior = prior
        self.prop = proposal
        self.lik = likelihood
        self.data = data
        self.res = {'SAMPLES':[], # ID
                    'ALPHAS':[],
                    'PARAMS':[], # ID
                    'PARAMS_PROPS': [],
                    'ACCEPT_INDEX':[],
                    'LIK':[],
                    'PRIOR':[],
                    'LIK_':[],
                    'PRIOR_':[],
                     }

        self.lookup = {} # dict of dicts
        self.time = 0.0
        self.iter = 0
        self.outfile = outfile
        self.last_params = None
        self.init = None

    def run(self, it=10000, fixed_init=None):
        tic = time.time()

        # Initialisation
        if fixed_init is not None:
            params = fixed_init
        else:
            params = self.prior.Sample()

        id_p = params.GetID()
        lik_p = self.lik.PDF(params)
        prior_p = self.prior.PDF(params)

        self.lookup[id_p] = {'LIK': lik_p}
        if params._tree and (self.prop._skip is not None):
            tree_id_p = params._tree.GetID()
        else:
            tree_id_p = ''
        self.init  = {'SAMPLE': id_p, 'LIK': lik_p, 'PRIOR': prior_p,
                      'BASIS_ID': ''.join(np.array(params._basis_active, dtype=str)), 'TREE_ID': tree_id_p}

        print(params)
        print("loglik: " + str(lik_p) + ", logprior: " + str(prior_p))

        # Iterate
        for i in tqdm(range(it)):
            assert(self.prop.counter == i)
            params_ = self.prop.Sample(params)

            id_p_ = params_.GetID()
            self.res['PARAMS'].append(id_p_)
            # Fetch likelihood if memoised
            if id_p_ in self.lookup.keys():
                lik_p_ = self.lookup[id_p_]['LIK']
            else:
                lik_p_ = self.lik.PDF(params_)
                self.lookup[id_p_] = {'LIK': lik_p_}

            prior_p_ = self.prior.PDF(params_)
            basis_id_p_ = ''.join(np.array(params_._basis_active, dtype=str))

            if params_._tree and (self.prop._skip is not None):
                if (self.prop.counter - 1) % self.prop._skip == 0:
                    tree_id_p_ = params_._tree.GetID()
            else:
                tree_id_p_ = ''

            lik_r = lik_p_ - lik_p
            prior_r = prior_p_ - prior_p
            prop_r = self.prop.PDF_ratio(params_)
            alpha = lik_r + prior_r + prop_r

            self.res['PARAMS_PROPS'].append({'PRIOR': prior_p_, 'BASIS_ID': basis_id_p_, 'TREE_ID': tree_id_p_, 'PROP_RATIO': prop_r})
            self.res['LIK_'].append(lik_p_)
            self.res['PRIOR_'].append(prior_p_)

            self.res['ALPHAS'].append(alpha)
            if np.log(np.random.uniform()) < alpha:
                self.res['ACCEPT_INDEX'].append(1)
                self.res['SAMPLES'].append(id_p_)
                self.res['LIK'].append(lik_p_)
                self.res['PRIOR'].append(prior_p_)
                id_p = id_p_
                params = params_
                lik_p = lik_p_
                prior_p = prior_p_
            else:
                self.res['ACCEPT_INDEX'].append(0)
                self.res['SAMPLES'].append(id_p)
                self.res['LIK'].append(lik_p)
                self.res['PRIOR'].append(prior_p)
                params = self.prop.Revert(params_)


        self.time = time.time() - tic # in seconds
        self.iter = it
        self.last_params = params.copy()
        return 0

    def get_summary(self, true_g, b, inc_distances=True, thin=100, acc_scaled_size=None):
        return MCMC_summary(self, true_g, b=b, inc_distances=inc_distances, thin=thin, acc_scaled_size=acc_scaled_size)

    def save_object(self):
        with open(self.outfile, 'wb') as handle:
            pickle.dump(self, handle)
        return 0

    def continue_chain(self, it):
        self.run(it, fixed_init= self.last_params)
        return 0

import numpy as np
from utils.diagnostics import IAC_time, str_list_to_adjm

class MCMC_summary():
    def __init__(self, sampler, true_g, b=0, alpha=.5, inc_distances=True, thin=100, acc_scaled_size=None):
        self.time = sampler.time
        self.iter = sampler.iter
        self.last_params = sampler.last_params
        self.likelihoods = sampler.res['LIK']
        self.posteriors = np.array(sampler.res['LIK'][b::thin]) + np.array(sampler.res['PRIOR'][b::thin])
        self.sizes = list(map(lambda s: np.sum(self._str_to_int_list(s)), sampler.res['SAMPLES'][b::thin]))
        self.bases = self._get_basis_ct(sampler)[b::thin]
        self.summary = self._get_summary(sampler, b, inc_distances=inc_distances, thin=thin, acc_scaled_size=acc_scaled_size)

        self.accuracies = [self._get_accuracies(true_g, self._get_median_graph(sampler, true_g, threshold, b=b, thin=thin)) \
                            for threshold in [.25, .5, .75]]
        self.AUCs = [self._get_AUCs(true_g, self._get_median_graph(sampler, true_g, threshold, b=b, thin=thin)) \
                            for threshold in [.25, .5, .75]]
        self.F1s = [self._get_F1s(true_g, self._get_median_graph(sampler, true_g, threshold, b=b, thin=thin)) \
                            for threshold in [.25, .5, .75]]

    def _get_median_graph(self, sampler, true_g, alpha=.5, b=0, thin=1):
        adjm = str_list_to_adjm(len(true_g), sampler.res['SAMPLES'][b::thin])
        return (adjm > alpha).astype(int)

    def _get_basis_ct(self, sampler):
        basis_ct = []
        if sampler.res['ACCEPT_INDEX'][0] == 0:
            basis_ct.append(np.sum(self._str_to_int_list(sampler.init['BASIS_ID'])))
        else:
            basis_ct.append(np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][0]['BASIS_ID'])))

        for i in range(1, len(sampler.res['ACCEPT_INDEX'])):
            if sampler.res['ACCEPT_INDEX'][i]:
                basis_ct.append(np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][i]['BASIS_ID'])))
            else:
                basis_ct.append(basis_ct[-1])
        return basis_ct

    def _get_accuracies(self, g, md):
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

    def _get_AUCs(self, g, md):
        from sklearn.metrics import roc_auc_score
        triu = np.triu_indices(len(g), 1)
        try:
            auc = roc_auc_score(g.GetBinaryL(), np.array(md[triu], dtype=bool))
        except ValueError:
            auc = np.nan
        return auc

    def _get_F1s(self, g, md):
        from sklearn.metrics import f1_score
        triu = np.triu_indices(len(g), 1)
        return f1_score(g.GetBinaryL(), np.array(md[triu], dtype=bool))

    def _str_to_int_list(self, s):
        return np.array(list(s), dtype=int)

    def _get_basis_ct(self, sampler):
        basis_ct = []
        if sampler.res['ACCEPT_INDEX'][0] == 0:
            basis_ct.append(np.sum(self._str_to_int_list(sampler.init['BASIS_ID'])))
        else:
            basis_ct.append(np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][0]['BASIS_ID'])))

        for i in range(1, len(sampler.res['ACCEPT_INDEX'])):
            if sampler.res['ACCEPT_INDEX'][i]:
                basis_ct.append(np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][i]['BASIS_ID'])))
            else:
                basis_ct.append(basis_ct[-1])
        return basis_ct

    def _get_distances(self, str_list, dist, values_to_save=5000):
        """diameter of a unique graph_id list according to specified dist: str * str -> int"""
        from itertools import combinations
        a = np.sort([ dist(s1, s2) for s1, s2 in combinations(str_list, 2) ])
        return a

    def _get_generalised_variance(self, str_list):
        X =  np.array([np.array(list(s), dtype=int) for s in str_list])
        X = X - np.mean(X, axis=0)
        cov = (np.transpose(X) @ X) / (X.shape[0] - 1)
        return np.linalg.det(cov)

    def _get_total_variance(self, str_list):
        X = np.array([np.array(list(s), dtype=int) for s in str_list])
        X = X - np.mean(X, axis=0)
        cov = (np.transpose(X) @ X) / (X.shape[0] - 1)
        return np.trace(cov)

    def _get_summary(self, sampler, b=0, inc_distances=True, thin=1, acc_scaled_size=None):
        posts = np.array(sampler.res['LIK'], dtype=float)[b::thin] + np.array(sampler.res['PRIOR'], dtype=float)[b::thin]
        posts_ = np.array(sampler.res['LIK_'], dtype=float)[b::thin] + np.array(sampler.res['PRIOR_'], dtype=float)[b::thin]
        sizes = list(map(lambda s: np.sum(self._str_to_int_list(s)), sampler.res['SAMPLES']))[b::thin]
        sizes_ = list(map(lambda s: np.sum(self._str_to_int_list(s)), sampler.res['PARAMS']))[b::thin]
        n_bases = self._get_basis_ct(sampler)[b::thin]
        n_bases_ = [np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][i]['BASIS_ID'])) for i in range(sampler.iter)][b::thin]

        trees = [pp['TREE_ID'] for pp in sampler.res['PARAMS_PROPS']]
        change_tree = np.where(list(map(lambda t, t_: t != t_, trees[:-1], trees[1:])))[0] + 1

        d = {}

        print('Calculating IATs...')
        d['IAT_posterior'] = IAC_time(posts)
        d['IAT_sizes'] = IAC_time(sizes)
        d['IAT_bases'] = IAC_time(n_bases)

        d['IAT_posterior_'] = IAC_time(posts_)
        d['IAT_sizes_'] = IAC_time(sizes_)
        d['IAT_bases_'] = IAC_time(n_bases_)

        d['accept_rate'] = np.sum(sampler.res['ACCEPT_INDEX']) / len(sampler.res['ACCEPT_INDEX'])

        n_bases_unthinned = self._get_basis_ct(sampler)[b:]
        n_bases_unthinned_ = [np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][i]['BASIS_ID'])) for i in range(sampler.iter)][b:]
        n_birth = np.sum((np.array(n_bases_unthinned)[1:] - np.array(n_bases_unthinned)[:-1]) == 1)
        d['accept_birth'] = n_birth / (n_birth + np.sum((np.array(n_bases_unthinned_) - np.array(n_bases_unthinned)) == 1))
        n_death = np.sum((np.array(n_bases_unthinned)[1:] - np.array(n_bases_unthinned)[:-1]) == -1)
        d['accept_death'] = n_death / (n_death + np.sum((np.array(n_bases_unthinned_) - np.array(n_bases_unthinned)) == -1))

        d['tree_accept_ct'] = len(set(change_tree).intersection(set(np.where(sampler.res['ACCEPT_INDEX'])[0])))
        d['max_posterior'] = np.max(posts)

        d['states_visited'] = len(np.unique(sampler.res['SAMPLES'][b::thin]))
        d['states_considered'] = len(np.unique(sampler.res['PARAMS'][b::thin]))

        if inc_distances:
            print('Calculating distances...')
            from utils.diagnostics import jaccard_distance, hamming_distance, size_distance

            thinned = sampler.res['SAMPLES'][b::thin]
            print(len(thinned))
            d['jaccard_distances'] = self._get_distances(thinned, jaccard_distance)
            d['hamming_distances'] = self._get_distances(thinned, hamming_distance)
            d['size_distances'] = self._get_distances(thinned, size_distance)

            thinned_ = sampler.res['PARAMS'][b::thin]
            d['jaccard_distances_'] = self._get_distances(thinned_, jaccard_distance)
            d['hamming_distances_'] = self._get_distances(thinned_, hamming_distance)
            d['size_distances_'] = self._get_distances(thinned_, size_distance)

            # uniq = np.unique(thinned)
            # d['jaccard_distances'] = self._get_distances(uniq, jaccard_distance)
            # d['hamming_distances'] = self._get_distances(uniq, hamming_distance)
            # d['size_distances'] = self._get_distances(uniq, size_distance)

            # uniq_ = np.unique(thinned_)
            # d['jaccard_distances_'] = self._get_distances(uniq_, jaccard_distance)
            # d['hamming_distances_'] = self._get_distances(uniq_, hamming_distance)
            # d['size_distances_'] = self._get_distances(uniq_, size_distance)

        print('Calculating variances...')
        d['gvar'] = self._get_generalised_variance(sampler.res['SAMPLES'][b::thin])
        d['gvar_'] = self._get_generalised_variance(sampler.res['PARAMS'][b::thin])

        d['tvar'] = self._get_total_variance(sampler.res['SAMPLES'][b::thin])
        d['tvar_'] = self._get_total_variance(sampler.res['PARAMS'][b::thin])

        if acc_scaled_size:
            print('Calculating acc_scaled distances...')
            print(f"acc_scaled_size: {acc_scaled_size}")
            cob_freq = sampler.prop._skip
            accept_idx = np.where(sampler.res['ACCEPT_INDEX'])[0]
            if cob_freq:
                accept_idx = accept_idx[accept_idx % cob_freq != 0]
            first_x_idx = accept_idx[:acc_scaled_size] + 1 # plus 1 for next proposed
            last_x_idx = accept_idx[-acc_scaled_size:] + 1

            print(first_x_idx.shape)
            print(last_x_idx.shape)

             # Edge case where the last iteration is accepted (+1 cause out of index)
            first_x_idx = first_x_idx[first_x_idx < sampler.iter]
            last_x_idx = last_x_idx[last_x_idx < sampler.iter]

            first_x = np.array(sampler.res['PARAMS'])[first_x_idx]
            last_x = np.array(sampler.res['PARAMS'])[last_x_idx]

            if inc_distances:
                from utils.diagnostics import jaccard_distance, hamming_distance, size_distance
                d['jaccard_distances_start'] = self._get_distances(first_x, jaccard_distance)
                d['jaccard_distances_end'] = self._get_distances(last_x, jaccard_distance)
                d['hamming_distances_start'] = self._get_distances(first_x, hamming_distance)
                d['hamming_distances_end'] = self._get_distances(last_x, hamming_distance)
                d['size_distances_start'] = self._get_distances(first_x, size_distance)
                d['size_distances_end'] = self._get_distances(last_x, size_distance)

            print('Calculating acc_scaled variances...')
            d['as_start_gvar'] = self._get_generalised_variance(first_x)
            d['as_end_gvar'] = self._get_generalised_variance(last_x)
            d['as_start_tvar'] = self._get_total_variance(first_x)
            d['as_end_tvar'] = self._get_total_variance(last_x)

            d['first_x_idx'] = first_x_idx
            d['last_x_idx'] = last_x_idx

        d['time'] = sampler.time

        return d


import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from utils.diagnostics import IAC_time, str_list_to_adjm


class MCMC_Summarizer():
    def __init__(self, files, burnin=[0, 5000, 15000], graphdir='data/', samplerdir= 'results/', outdir='results/'):
        self.files = files
        self.burnin = burnin
        self.graphdir = graphdir
        self.samplerdir = samplerdir
        self.outdir = outdir

    def _get_accuracies(self, g, md):
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

    def _plot_traces(self, a, title, outfile):
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        fig = plt.figure(figsize=(20,10))
        plt.plot(a)
        plt.title(title, fontsize=30)
        plt.xlabel('Number of Iterations', fontsize=20)
        fig.savefig(outfile)
        plt.close(fig)
        return 0

    def _str_to_int_list(self, s):
        return np.array(list(s), dtype=int)

    def _get_basis_ct(self, sampler):
        basis_ct = []
        if sampler.res['ACCEPT_INDEX'][0] == 0:
            basis_ct.append(np.sum(self._str_to_int_list(sampler.init['BASIS_ID'])))
        else:
            basis_ct.append(np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][0]['BASIS_ID'])))

        for i in range(1, len(sampler.res['ACCEPT_INDEX'])):
            if sampler.res['ACCEPT_INDEX'][i]:
                basis_ct.append(np.sum(self._str_to_int_list(sampler.res['PARAMS_PROPS'][i]['BASIS_ID'])))
            else:
                basis_ct.append(basis_ct[-1])
        return basis_ct

    def summarize(self):
        l = ['basis', 'graph', 'n', 'iter', 'time',
            'accept_rate', 'tree_accept_ct', 'max_posterior', 'states_visited',
            'IAT_posterior', 'IAT_sizes', 'IAT_bases',
            'TP', 'TN', 'FP', 'FN']

        for b in self.burnin:
            d = {k:[] for k in l}
            for f in self.files:
                with open(self.samplerdir + f, 'rb') as handle:
                    sampler = pickle.load(handle)

                # Information about the experiments
                b_str, g_str = f.split('_')[:2] # basis and graph names
                n = int("".join(filter(str.isdigit, os.getcwd().split('/')[-1])))
                d['basis'].append(b_str)
                d['graph'].append(g_str)
                d['n'].append(n)
                d['iter'].append(sampler.iter)
                d['time'].append(sampler.time)


                # Mixing Performance
                posts = np.array(sampler.res['LIK'], dtype=float)[b:] + np.array(sampler.res['PRIOR'], dtype=float)[b:]
                sizes = list(map(lambda s: np.sum(self._str_to_int_list(s)), sampler.res['SAMPLES']))[b:]
                n_bases = self._get_basis_ct(sampler)[b:]
                trees = [pp['TREE_ID'] for pp in sampler.res['PARAMS_PROPS']]
                change_tree = np.where(list(map(lambda t, t_: t != t_, trees[:-1], trees[1:])))[0] + 1

                dirname = self.outdir + 'vis-burnin-' + str(b) + '/'
                self._plot_traces(posts, 'Log Posterior', dirname + 'post_traces/' + b_str + '_' + g_str + '_post_trace.pdf')
                self._plot_traces(sizes, 'Number of Edges', dirname + 'size_traces/' + b_str + '_' + g_str + '_size_trace.pdf')
                self._plot_traces(n_bases, 'Number of Bases', dirname + 'basis_traces/' + b_str + '_' + g_str + '_basisct_trace.pdf')

                d['IAT_posterior'].append(IAC_time(posts))
                d['IAT_sizes'].append(IAC_time(sizes))
                d['IAT_bases'].append(IAC_time(n_bases))

                d['accept_rate'].append(np.sum(sampler.res['ACCEPT_INDEX']) / len(sampler.res['ACCEPT_INDEX']))
                d['tree_accept_ct'].append(len(set(change_tree).intersection(set(np.where(sampler.res['ACCEPT_INDEX'])[0]))))
                d['max_posterior'].append(np.max(posts))
                d['states_visited'].append(len(np.unique(sampler.res['SAMPLES'][b:])))


                # Accuracy Performance
                infile = self.graphdir + 'graph_' + d['graph'][-1] + '.pkl'
                with open(infile, 'rb') as handle:
                    g = pickle.load(handle)

                assert(n == len(g))

                adjm = str_list_to_adjm(len(g), sampler.res['SAMPLES'][b:])
                with open(dirname + b_str  + '_' + g_str + '_adjm.pkl', 'wb') as handle:
                    pickle.dump(adjm, handle)

                median_g = (adjm > .75).astype(int)

                TP, TN, FP, FN = self._get_accuracies(g, median_g)

                d['TP'].append(TP)
                d['TN'].append(TN)
                d['FP'].append(FP)
                d['FN'].append(FN)

            df = pd.DataFrame(d)
            df.to_csv( self.outdir + 'combined_summary-burnin-' + str(b) + '.csv', index=False)

