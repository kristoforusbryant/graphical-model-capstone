import numpy as np
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

    def run(self, it=7500, fixed_init=None):
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

            self.res['PARAMS_PROPS'].append({'PRIOR': prior_p_, 'BASIS_ID': basis_id_p_, 'TREE_ID': tree_id_p_})

            self.res['LIK_'].append(lik_p_)
            self.res['PRIOR_'].append(prior_p_)

            lik_r = lik_p_ - lik_p
            prior_r = prior_p_ - prior_p
            alpha = lik_r + prior_r

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

    def save_object(self):
        with open(self.outfile, 'wb') as handle:
            pickle.dump(self, handle)
        return 0

    def continue_chain(self, it):
        self.run(it, fixed_init= self.last_params)
        return 0

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
                n = int(os.getcwd().split('/')[-1].replace('nodes', ''))
                d['basis'].append(b_str)
                d['graph'].append(g_str)
                d['n'].append(n)
                d['iter'].append(sampler.iter)
                d['time'].append(sampler.time)


                # Mixing Performance
                posts = np.array(sampler.res['LIK'], dtype=float)[b:] + np.array(sampler.res['PRIOR'], dtype=float)[b:]
                sizes = list(map(lambda s: np.sum(self._str_to_int_list(s)), sampler.res['SAMPLES']))[b:]
                n_bases = list(map(lambda pp: np.sum(self._str_to_int_list(pp['BASIS_ID'])), sampler.res['PARAMS_PROPS']))[b:]
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

                median_g = (adjm > .5).astype(int)

                TP, TN, FP, FN = self._get_accuracies(g, median_g)

                d['TP'].append(TP)
                d['TN'].append(TN)
                d['FP'].append(FP)
                d['FN'].append(FN)

            df = pd.DataFrame(d)
            df.to_csv( self.outdir + 'combined_summary-burnin-' + str(b) + '.csv', index=False)

