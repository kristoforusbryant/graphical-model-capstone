import numpy as np
import scipy.stats as st
from tqdm import tqdm
import copy
import time
import warnings
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # to not display plots
from utils.diagnostics import ACF, IAC_time

class MCMC_Sampler: 
    def __init__(self, prior, proposal, likelihood, data, reps=1, outdir=""):
        self.prior = prior 
        self.prop = proposal 
        self.lik = likelihood
        self.data = data 
        self.res = [{'SAMPLES':[], # ID
                    'ALPHAS':[],
                    'PARAMS':[], # ID
                    'ACCEPT_INDEX':[], 
                    'LIK':[], 
                    'PRIOR':[], 
                    'PROP':[]} for _ in range(reps)]
        
        self.lookup = {} # dict of dicts 
        
        self.time = [0.0 for _ in range(reps)]
        self.iter = [0 for _ in range(reps)]
        self.summary = [{} for _ in range (reps)]
        self.reps = reps
        self.outdir = outdir
        
    def Run(self, it=7500, summarize=False, trackProposed=False, saveEach=False, fixed_init=None): 
        tic = time.time()
        
        if trackProposed: 
            for rep in range(self.reps): 
                self.res[rep]['LIK_'] = []
                self.res[rep]['PRIOR_'] = []
                self.res[rep]['PROP_'] = []
        
        # Sampler
        for rep in range(self.reps): 
            # Initialisation 
            if fixed_init is not None: 
                params = fixed_init
            else: 
                params = self.prior.Sample()
                
            id_p = params.GetID()
            if id_p in self.lookup.keys(): 
                lik_p = self.lookup[id_p]['LIK']
                prior_p = self.lookup[id_p]['PRIOR']
            else:
                lik_p = self.lik.PDF(params)
                prior_p = self.prior.PDF(params)
                basis_id_p = ''.join(params._basis_active.astype(int).astype(str).tolist())
                self.lookup[id_p] = {'LIK': lik_p, 'PRIOR': prior_p,
                                     'BASIS_ID': basis_id_p}
                                     #'OBJ': params}
            
            print(params)
            print("loglik: " + str(lik_p) + ", logprior: " + str(prior_p))
            for i in tqdm(range(it)):
                params_ = self.prop.Sample(params)
                
                id_p_ = params_.GetID()
                self.res[rep]['PARAMS'].append(id_p_)
                if id_p_ in self.lookup.keys(): 
                    lik_p_ = self.lookup[id_p_]['LIK']
                    prior_p_ = self.lookup[id_p_]['PRIOR']
                else:
                    lik_p_ = self.lik.PDF(params_)
                    prior_p_ = self.prior.PDF(params_)
                    basis_id_p_ = ''.join(params_._basis_active.astype(int).astype(str).tolist())
                    self.lookup[id_p_] = {'LIK': lik_p_, 'PRIOR': prior_p_, 
                                          'BASIS_ID': basis_id_p_}
                                          #'OBJ': params_}
                
                prop_p_ = self.prop.PDF(params_, params ) # params_ -> params
                prop_p = self.prop.PDF(params, params_) # params -> params_
                
                if trackProposed: 
                    self.res[rep]['LIK_'].append(lik_p_) 
                    self.res[rep]['PRIOR_'].append(prior_p_)
                    self.res[rep]['PROP_'].append((prop_p, prop_p_))

                lik_r = lik_p_ - lik_p 
                prior_r = prior_p_ - prior_p 
                prop_r = prop_p_ - prop_p

                alpha = lik_r + prior_r + prop_r 

                self.res[rep]['ALPHAS'].append(alpha)
                if np.log(np.random.uniform()) < alpha:
                    self.res[rep]['ACCEPT_INDEX'].append(1)
                    self.res[rep]['SAMPLES'].append(id_p_)
                    self.res[rep]['LIK'].append(lik_p_) 
                    self.res[rep]['PRIOR'].append(prior_p_)
                    self.res[rep]['PROP'].append((prop_p, prop_p_))
                    id_p = id_p_
                    params = params_ 
                    lik_p = lik_p_ 
                    prior_p = prior_p_
                else:
                    self.res[rep]['ACCEPT_INDEX'].append(0)
                    self.res[rep]['SAMPLES'].append(id_p)
                    self.res[rep]['LIK'].append(lik_p) 
                    self.res[rep]['PRIOR'].append(prior_p)
                    self.res[rep]['PROP'].append((prop_p_, prop_p))

            self.time[rep] = time.time() - tic # in seconds 
            self.iter[rep] = it
            
            if saveEach: 
                with open(self.outdir + 'rep' + str(rep) + '.pkl', 'wb') as handle:
                    pickle.dump(self.res[rep], handle)
                self.res[rep] = {'SAMPLES':[],
                                'ALPHAS':[],
                                'PARAMS':[], 
                                'ACCEPT_INDEX':[], 
                                'LIK':[], 
                                'PRIOR':[], 
                                'PROP':[]}
        if summarize: 
            self.Summarize()

        return 0 
    
    def SaveRaw(self, outfile): 
        with open(outfile, 'wb') as handle:
            pickle.dump((self.res, self.lookup), handle)
            
    def SaveSummary(self, outfile):
        with open(outfile, 'wb') as handle:
            pickle.dump(self.summary, handle)
    
    def Summarize(self):
        for rep in range(self.reps): 
            size = [np.sum(np.array(list(p_id), dtype=int)) for p_id in self.res[rep]['SAMPLES'] ]
            posterior = np.array(self.res[rep]['LIK']) + np.array(self.res[rep]['PRIOR'])

            uniq = {} #k: graph strings, v: (count, size)
            for p_id in self.res[rep]['SAMPLES']: 
                if p_id not in uniq.keys():
                    uniq[p_id] = [1, np.sum(np.array(list(p_id), dtype=int))]
                else: 
                    uniq[p_id][0] += 1

            uniq = np.array(list(uniq.values()))
            uniq = uniq[np.argsort(uniq[:, 1])] 

            self.summary[rep]['ACCEPT_RATE'] = np.sum(self.res[rep]['ACCEPT_INDEX']) / len(self.res[rep]['ACCEPT_INDEX'])
            self.summary[rep]['MAX_LOGPOSTERIOR'] = np.max(posterior)
            self.summary[rep]['IAC_TIME_POSTERIOR'] = IAC_time(posterior, M=1000)
            self.summary[rep]['IAC_TIME_SIZE'] = IAC_time(size, M=1000)
            self.summary[rep]['STATES_VISITED'] = len(uniq)
            self.summary[rep]['TIME'] = self.time[rep]
            self.summary[rep]['ITER'] = len(self.res[rep]['SAMPLES'])
    
    
    def GetTrace(self, dof, outfile, additional_dicts = None, list_first=True, burnin=0): # TODO: thinning
        assert self.reps == len(additional_dicts), "reps and number of additional dicts does not match"
        
        # dof is a dictionary of functions from one element of MCMC_Sampler.res to floats
        lod = []
        if additional_dicts is not None and list_first:
            for rep in range(self.reps):
                d = additional_dicts[rep]
                for k,f in dof.items(): 
                    d[k] = [f(p_id) for p_id in self.res[rep]['SAMPLES'][burnin:]]
                lod.append(d)
        elif additional_dicts is not None and not list_first: 
            for rep in range(self.reps):
                d = {dof[k]: [f(p_id) for p_id in self.res[rep]['SAMPLES'][burnin:]] for k,f in dof.items()}
                for k,v in additional_dict: 
                    d[k] = v
                lod.append(d)
        else: 
            for rep in range(self.reps):
                d = {dof[k]: [f(p_id) for p in self.res[rep]['SAMPLES'][burnin:]] for k,f in dof.items()}
                lod.append(d)

        rownames = list(lod[0].keys())
                
        # SETTING UP GRIDS FOR PLOTTING
        if self.reps > 10:
            warnings.warn("large number of repetition, each subplot may be too small to read")
            
        if self.reps == 1: 
            fig, axs = plt.subplots(len(rownames), self.reps, figsize=(25,25), sharex='col', sharey='row')
            fig.tight_layout()
            for i in range(len(rownames)):
                axs[i,].xaxis.set_tick_params(labelsize=15)
                axs[i,].yaxis.set_tick_params(labelsize=15)
                axs[i,].set_ylabel(rownames[i], fontsize=20)
                
                vals = list(lod[0].values())
                axs[i,].plot(vals[i])
                
            fig.savefig(outfile, dpi=250, bbox_inches='tight')
            return fig
            
        else: 
            fig, axs = plt.subplots(len(rownames), self.reps, figsize=(25,25), sharex='col', sharey='row')
            fig.tight_layout()
            for i in range(len(rownames)): 
                for j in range(self.reps):
                    axs[i,j].xaxis.set_tick_params(labelsize=15)
                    axs[i,j].yaxis.set_tick_params(labelsize=15)

            for i in range(self.reps): 
                axs[0,i].set_title("rep" + str(i), fontsize=20)
            for i in range(len(rownames)): 
                axs[i,0].set_ylabel(rownames[i], fontsize=20)

            # FOR EVERY TYPE 
            for rep in range(self.reps):
                ctr = 0
                for v in lod[rep].values():    
                    axs[ctr, rep].plot(v)
                    ctr = ctr + 1
            fig.savefig(outfile, dpi=250, bbox_inches='tight')
            return fig         