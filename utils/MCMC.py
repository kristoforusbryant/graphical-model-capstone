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
