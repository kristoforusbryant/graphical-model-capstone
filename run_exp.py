import os, sys
import numpy as np
import pickle
import json
from utils.diagnostics import create_edge_matrix
import PIL
import matplotlib.pyplot as plt
from utils.Graph import Graph
from utils.laplace_approximation import constrained_cov
from utils.G_Wishart import G_Wishart

# Get Input project dir 
exppath = str(sys.argv[1]) 
# exppath = "experiments/five_on_five"

### Read config file 
filename=os.path.join(exppath, "config.json")
with open(filename) as handle: 
    CONFIG = json.load(handle)     
    
### Generate Params and Data if necessary inputs does not exits 
if not all(os.path.exists(exppath+"/"+obj) for obj in ["Params.pkl", "Data.pkl"]): 
    print("Generating using experiments/five_on_five/generator.py")
    os.system('python ' + exppath + '/generator.py')
    
### Import Prior, Proposal, and Likelihood Classes 
import importlib
Prior = importlib.import_module("dists.priors." + CONFIG['PRIOR']).Prior
Proposal = importlib.import_module("dists.proposals." + CONFIG['PROPOSAL']).Proposal
Likelihood = importlib.import_module("dists.likelihoods." + CONFIG['LIKELIHOOD']).Likelihood

### Load Params and Data 
filename = os.path.join(exppath, 'Params.pkl')
with open(filename, 'rb') as handle:
    PARAMS = pickle.load(handle)

filename = os.path.join(exppath, 'Data.pkl')
with open(filename, 'rb') as handle:
    DATA = pickle.load(handle)
    
### Load Hyperparams 
if "COUNTPRIOR" in list(CONFIG.keys()): 
    CountPrior = importlib.import_module("dists.count_priors." + CONFIG['COUNTPRIOR']["name"]).CountPrior
    prob_c = CountPrior(CONFIG['COUNTPRIOR']["params"])
    
if "SIZEPRIOR" in list(CONFIG.keys()):     
    SizePrior = importlib.import_module("dists.size_priors." + CONFIG['SIZEPRIOR']["name"]).SizePrior 
    prob_s = SizePrior(CONFIG['SIZEPRIOR']["params"])
    
    
### Run the MCMC 
from utils.MCMC import MCMC_Sampler
from multiprocessing import Pool 

def OneThread(i):
    n = DATA[i].shape[1]
    reps = CONFIG["N_REPS"][i]
    
    if "TREEPRIOR" in list(CONFIG.keys()): 
        TreePrior = importlib.import_module("dists.spanning_tree_priors." + CONFIG['TREEPRIOR']).STPrior  
        tree_prior = TreePrior(n)
    
    # Initialise prior, prop, lik, data 
    if CONFIG['PRIOR'] in ["uniform", "basis_uniform"]: 
        prior = Prior(n, PARAMS[i].__class__, basis=PARAMS[i]._basis)
    if CONFIG['PRIOR'] in ["basis_count_size"]: 
        prior = Prior(n, PARAMS[i].__class__, basis=PARAMS[i]._basis, prob_c=prob_c , prob_s=prob_s)
    if CONFIG['PRIOR'] in ["basis_count_size_t"]: 
        prior = Prior(n, PARAMS[i].__class__, prob_c=prob_c, prob_s=prob_s, tree_prior=tree_prior)
    if CONFIG['PROPOSAL'] in ["uniform", "basis_uniform"]: 
        prop = Proposal(n, PARAMS[i].__class__)
    if CONFIG['PROPOSAL'] in ["basis_size", "basis_size_bd"]: 
        prop = Proposal(n, PARAMS[i].__class__, basis=PARAMS[i]._basis, prob_s=prob_s)
    if CONFIG['PROPOSAL'] in ["basis_size_t", "basis_size_bd_t"]: 
        prop = Proposal(n, PARAMS[i].__class__, prob_s, tree_prior=tree_prior, skip=CONFIG['SKIP'])
    
    delta = 3 
    D = np.eye(n) # (delta, D) hyperpriors
    lik = Likelihood(DATA[i], 3, D, PARAMS[i].__class__)
    
    sampler = MCMC_Sampler(prior, prop, lik, DATA[i], reps=reps)
    if CONFIG["FIXED_INIT"]:
        sampler.Run(CONFIG['ITER'][i], summarize=True, trackProposed=True, fixed_init=PARAMS[i])
    else: 
        sampler.Run(CONFIG['ITER'][i], summarize=True, trackProposed=True)
    
    sampler.SaveRaw(os.path.join(exppath, 'res/raw_'+str(i)+'.pkl'))
    sampler.SaveSummary(os.path.join(exppath, 'summary/summary_'+str(i)+'.pkl'))
    
p = Pool()
p.map(OneThread, range(len(DATA)))

### Plotting
print("Plotting using experiments/five_on_five/plotter.py")
os.system('python ' + exppath + '/plotter.py')
    
