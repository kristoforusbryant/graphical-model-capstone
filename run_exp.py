import os, sys
import numpy as np
import pickle
import json

# Get Input project dir 
exppath = str(sys.argv[1]) 

### Read config file 
filename=os.path.join(exppath, "config.json")
with open(filename) as handle: 
    CONFIG = json.load(handle) 
    
### Generate Params and Data if necessary inputs does not exits 
if not all(os.path.exists(exppath+"/"+obj) for obj in ["Params.pkl", "Data.pkl"]): 
    print("Generating using " + exppath + "/generator.py ...")
    os.system('python ' + exppath + '/generator.py')
    
### Import Prior, Proposal, and Likelihood Classes 
import importlib
Prior = importlib.import_module("dists.priors." + CONFIG['PRIOR']).Prior
Proposal = importlib.import_module("dists.proposals." + CONFIG['PROPOSAL']).Proposal
Likelihood = importlib.import_module("dists.likelihoods." + CONFIG['LIKELIHOOD']).Likelihood

assert Prior.ParamType() == Proposal.ParamType(), "Prior and Proposal have different Parameter Types"
assert Prior.ParamType() == Likelihood.ParamType(), "Prior and Proposal have different Parameter Types"

### Load Params and Data 
filename = os.path.join(exppath, 'Params.pkl')
with open(filename, 'rb') as handle:
    PARAMS = pickle.load(handle)

filename = os.path.join(exppath, 'Data.pkl')
with open(filename, 'rb') as handle:
    DATA = pickle.load(handle)
    
### Run the MCMC 
from utils.MCMC import MCMC_Sampler
for i in range(len(DATA)): 
    n = DATA[i].shape[1]
    reps = CONFIG["N_REPS"][i]
    
    # Initialise prior, prop, lik, data 
    prior = Prior(n)
    prop = Proposal(n)
    delta = 3 
    D = np.eye(n) # (delta, D) hyperpriors
    lik = Likelihood(DATA[i], 3, D)
    
    sampler = MCMC_Sampler(prior, prop, lik, DATA[i], reps=reps)
    sampler.Run(CONFIG['ITER'][i], summarize=True)
    
    sampler.SaveRaw(os.path.join(exppath, 'res_raw.pkl'))
    sampler.SaveSummary(os.path.join(exppath, 'summary.pkl'))

    # Define the statistics to be traced
    triu = np.triu_indices(n, 1) # upper-tri indices 
    triu_edge_list = [(triu[0][i], triu[1][i]) for i in range(len(triu[0]))]
    dof = {"(" + str(i) + "," + str(j) + ")" : lambda x,i=i,j=j: x.IsEdge(i, j) 
           for i,j in triu_edge_list}
    
    # Plot Traces
    fig = sampler.GetTrace(dof, os.path.join(exppath, 'vis') + "/trace" + str(i) + '.png', burnin=0) # CONFIG['BURNIN'][i]