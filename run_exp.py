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
    
### Run the MCMC 
from utils.MCMC import MCMC_Sampler
from multiprocessing import Pool 

def OneThread(i):
    n = DATA[i].shape[1]
    reps = CONFIG["N_REPS"][i]
    
    # Initialise prior, prop, lik, data 
    prior = Prior(n, PARAMS[i].__class__, basis=PARAMS[i]._basis)
    prop = Proposal(n, PARAMS[i].__class__)
    delta = 3 
    D = np.eye(n) # (delta, D) hyperpriors
    lik = Likelihood(DATA[i], 3, D, PARAMS[i].__class__)
    
    sampler = MCMC_Sampler(prior, prop, lik, DATA[i], reps=reps)
    sampler.Run(CONFIG['ITER'][i], summarize=True)
    
    sampler.SaveRaw(os.path.join(exppath, 'res/raw_'+str(i)+'.pkl'))
    sampler.SaveSummary(os.path.join(exppath, 'summary/summary_'+str(i)+'.pkl'))
    
p = Pool()
p.map(OneThread, range(len(DATA)))

### Plotting
print("Plotting using experiments/five_on_five/plotter.py")
os.system('python ' + exppath + '/plotter.py')
    
