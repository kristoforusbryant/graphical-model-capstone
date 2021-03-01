import sys, os
from os.path import dirname
filepath = dirname(os.path.realpath(__file__)) # adding file directory 
sys.path.append(filepath)
projectpath = dirname(dirname(filepath)) # adding directory containing utils
sys.path.append(projectpath)

import numpy as np
import pickle
import json
from utils.diagnostics import create_edge_matrix_from_binary_str
import PIL
import matplotlib.pyplot as plt
from utils.Graph import Graph
from utils.laplace_approximation import constrained_cov
from utils.G_Wishart import G_Wishart


# Reading Config file 
filename=os.path.join(filepath, "config.json")
with open(filename) as f: 
    CONFIG = json.load(f)

N_graphs = len(CONFIG['N_NODES'])
assert N_graphs == len(CONFIG['N_REPS']), "len(nodes) != len(reps)"
assert N_graphs == len(CONFIG['N_DATA']), "len(nodes) != len(data)"
assert N_graphs == len(CONFIG['ITER']), "len(nodes) != len(iter)"
assert N_graphs == len(CONFIG['BURNIN']), "len(nodes) != len(burnin)"

### Load Params and Data 
filename = os.path.join(filepath, 'Params.pkl')
with open(filename, 'rb') as handle:
    PARAMS = pickle.load(handle)

filename = os.path.join(filepath, 'Data.pkl')
with open(filename, 'rb') as handle:
    DATA = pickle.load(handle)

### Import Prior, Proposal, and Likelihood Classes 
import importlib
Prior = importlib.import_module("dists.priors." + CONFIG['PRIOR']).Prior
Proposal = importlib.import_module("dists.proposals." + CONFIG['PROPOSAL']).Proposal
Likelihood = importlib.import_module("dists.likelihoods." + CONFIG['LIKELIHOOD']).Likelihood

### Load Hyperparams 
if "COUNTPRIOR" in list(CONFIG.keys()): 
    CountPrior = importlib.import_module("dists.count_priors." + CONFIG['COUNTPRIOR']["name"]).CountPrior
    prob_c = CountPrior(CONFIG['COUNTPRIOR']["params"])
    
if "SIZEPRIOR" in list(CONFIG.keys()):     
    SizePrior = importlib.import_module("dists.size_priors." + CONFIG['SIZEPRIOR']["name"]).SizePrior 
    prob_s = SizePrior(CONFIG['SIZEPRIOR']["params"])

### Plotting 
from utils.MCMC import MCMC_Sampler
from multiprocessing import Pool 

def OnePlot(i):
    print(str(i) +  ': STARTING')
    import numpy as np 
    
    n = DATA[i].shape[1]
    reps = CONFIG["N_REPS"][i]
    # Initialise prior, prop, lik, data 
    if CONFIG['PRIOR'] in ["uniform", "basis_uniform"]: 
        prior = Prior(n, PARAMS[i].__class__, basis=PARAMS[i]._basis)
    if CONFIG['PRIOR'] in ["basis_count_size"]: 
        prior = Prior(n, PARAMS[i].__class__, basis=PARAMS[i]._basis, prob_c=prob_c , prob_s=prob_s)
    if CONFIG['PRIOR'] in ["basis_count_size_t"]: 
        prior = Prior(n, PARAMS[i].__class__, prob_c=prob_c, prob_s=prob_s)
    if CONFIG['PROPOSAL'] in ["uniform", "basis_uniform"]: 
        prop = Proposal(n, PARAMS[i].__class__)
    if CONFIG['PROPOSAL'] in ["basis_size", "basis_size_bd"]: 
        prop = Proposal(n, PARAMS[i].__class__, basis=PARAMS[i]._basis, prob_s=prob_s)
    if CONFIG['PROPOSAL'] in ["basis_size_t", "basis_size_bd_t"]: 
        prop = Proposal(n, PARAMS[i].__class__, prob_s)
        
    delta = 3 
    D = np.eye(n) # (delta, D) hyperpriors
    lik = Likelihood(DATA[i], 3, D, PARAMS[i].__class__)
    
    print(str(i) +  ': LOADING')
    filename = os.path.join(filepath, 'res/raw_'+str(i)+'.pkl')
    with open(filename, 'rb') as handle: 
        res, lookup = pickle.load(handle) 
    print(str(i) +  ': LOADED')
    
    sampler = MCMC_Sampler(prior, prop, lik, DATA[i], reps=reps)
    sampler.res = res
    sampler.lookup = lookup 
    
    # Define the statistics to be traced
    burnin = CONFIG['BURNIN'][i]
    truncate = 8
    
    sum_bin_str = lambda x: np.sum(np.array(list(x), dtype=int))
    basis = PARAMS[i]._basis
    
    # Define the statistics to be traced
    dof = {"size": lambda x: sum_bin_str(x),
           "basis_count": lambda x: sum_bin_str(sampler.lookup[x]['BASIS_ID']),
           "avg_basis_size": lambda x: np.mean([basis[i].EdgeCount() for i in range(len(basis)) 
                                                if x[i] == '1'])
          }
    add_dicts = []
    for rep in range(reps):
        d = {
            "lik": np.array(sampler.res[rep]['LIK'][burnin:]),
            "lik_prop": np.array(sampler.res[rep]['LIK_'][burnin:]), 
            "prior": np.array(sampler.res[rep]['PRIOR'][burnin:]),
            "prior_prop": np.array(sampler.res[rep]['PRIOR_'][burnin:]) , 
            "posterior": np.array(sampler.res[rep]['LIK'][burnin:]) + np.array(sampler.res[rep]['PRIOR'][burnin:]),
            "posterior_prop": np.array(sampler.res[rep]['LIK_'][burnin:]) + np.array(sampler.res[rep]['PRIOR_'][burnin:]) 
        }
        add_dicts.append(d)
        
    print(str(i) +  ': PLOTTING TRACES')
    # Plot Traces
    fig = sampler.GetTrace(dof, os.path.join(filepath, 'vis' + "/trace" + str(i) + '.png'), 
                           additional_dicts = add_dicts, list_first=True, burnin=burnin) 
    
    # Plot True and Preidcted Graphs
    AdjM_list = [create_edge_matrix_from_binary_str(rep['SAMPLES'], n) for rep in sampler.res]
    AdjM = sum(AdjM_list) / len(AdjM_list)
    AdjM[AdjM >= .5] = 1
    AdjM[AdjM < .5] = 0
    
    g = Graph(n)
    g.SetFromAdjM(AdjM)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    pos = g.GetCirclePos()
    PARAMS[i].Draw(ax = axes[0], pos=pos)
    axes[0].set_title('true', fontsize=20)
    g.Draw(ax = axes[1], pos=pos) 
    axes[1].set_title('predicted', fontsize=20)
    
    # Plot decreasing likelihoods (change ncols in the axes to 3 if we want to do this)
    data = DATA[i]
    U = data.transpose() @ data 

    # Get True Graph
    true_g = PARAMS[i]
    D_star = constrained_cov(true_g.GetDOL(), D + U, np.eye(n))
    GW_prior = G_Wishart(true_g, delta, D)
    GW_postr = G_Wishart(true_g, delta + data.shape[0], D_star)
    true_loglik = GW_postr.IG() - GW_prior.IG()

    import pandas as pd 
    import numpy as np 
    df = pd.DataFrame(res[0])
    df.sort_values('LIK', inplace=True, ascending=False)
    logliks = list(df.groupby('SAMPLES').head(1)['LIK'])
    
    axes[2].plot(logliks[:50])
    axes[2].hlines(true_loglik, 0, 50)

    axes[2].set_title('largest likelihoods', fontsize=20)
    
    fig.tight_layout()
    
    filename = os.path.join(filepath, 'vis/' + "Header"+str(i)+".png")
    fig.savefig(filename, dpi=250, bbox_inches='tight')
    
    
    print(str(i) +  ': COMBINING')
    #Combining with trace plot
    list_im = [os.path.join(filepath, 'vis/' + im) for im in ["Header"+str(i)+".png", 'trace'+str(i)+'.png']]
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    new_im = PIL.Image.new(imgs[0].mode, (imgs[1].size[0], imgs[0].size[1]))
    new_im.paste(imgs[0])

    imgs_comb = np.vstack( [new_im, imgs[1]] )
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    
    imgs_comb.save(os.path.join(filepath,'vis/'+'vis'+str(i)+'.png'))
    
p = Pool()
p.map(OnePlot, range(len(DATA)))

#for i in range(5): 
#    OnePlot(i)

