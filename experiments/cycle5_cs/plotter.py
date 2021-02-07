import sys, os
from os.path import dirname
filepath = dirname(os.path.realpath(__file__)) # adding file directory 
sys.path.append(filepath)
projectpath = dirname(dirname(filepath)) # adding directory containing utils
sys.path.append(projectpath)

import numpy as np
import pickle
import json
from utils.diagnostics import create_edge_matrix
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

### Plotting 
from utils.MCMC import MCMC_Sampler
from multiprocessing import Pool 

def OnePlot(i):
    n = DATA[i].shape[1]
    reps = CONFIG["N_REPS"][i]
    
    # Initialise prior, prop, lik, data 
    prior = Prior(n, PARAMS[i].__class__, basis=PARAMS[i]._basis)
    prop = Proposal(n, PARAMS[i].__class__)
    delta = 3 
    D = np.eye(n) # (delta, D) hyperpriors
    lik = Likelihood(DATA[i], 3, D, PARAMS[i].__class__)
    
    filename = os.path.join(filepath, 'res/raw_'+str(i)+'.pkl')
    with open(filename, 'rb') as handle: 
        res = pickle.load(handle) 
    
    sampler = MCMC_Sampler(prior, prop, lik, DATA[i], reps=reps)
    sampler.res = res
    
    # Define the statistics to be traced
    burnin = CONFIG['BURNIN'][i]
    truncate = 8
    
    # Define the statistics to be traced
    triu = np.triu_indices(n, 1) # upper-tri indices 
    triu_edge_list = [(triu[0][i], triu[1][i]) for i in range(len(triu[0]))]
    dof = {"(" + str(i) + "," + str(j) + ")" : lambda x,i=i,j=j: x.IsEdge(i, j) 
           for i,j in triu_edge_list}
    
    add_dicts = []
    for rep in range(reps):
        d = {
            "posterior": np.array(sampler.res[rep]['LIK'][burnin:]) + np.array(sampler.res[rep]['PRIOR'][burnin:]),
            "posterior_prop": np.array(sampler.res[rep]['LIK_'][burnin:]) + np.array(sampler.res[rep]['PRIOR_'][burnin:])
        }
        add_dicts.append(d)
    
    # Plot Traces
    fig = sampler.GetTrace(dof, 'vis' + "/trace" + str(i) + '.png', 
                           additional_dicts = add_dicts, list_first=True, burnin=burnin) # CONFIG['BURNIN'][i]
    
    # Plot True and Preidcted Graphs
    AdjM_list = [create_edge_matrix(rep['SAMPLES']) for rep in sampler.res]
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
    res = {'IG_prior': [],'IG_postr': []}

    all_graphs = Graph.get_all_graphs(n)
    for g in all_graphs:
        D_star = constrained_cov(g.GetDOL(), D + U, np.eye(n))
        GW_prior = G_Wishart(g, delta, D)
        GW_postr = G_Wishart(g, delta + data.shape[0], D_star)
        res['IG_prior'].append(GW_prior.IG())
        res['IG_postr'].append(GW_postr.IG())
    
    import pandas as pd 
    df = pd.DataFrame(res)
    df['loglik'] = df['IG_postr'] - df['IG_prior'] 
    df.sort_values(by='loglik', ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df.loglik[:50].plot.line(ax = axes[2])
    axes[2].set_title('largest likelihoods', fontsize=20)
    
    fig.tight_layout()
    
    filename = 'vis/' + "Header"+str(i)+".png"
    fig.savefig(filename, dpi=250, bbox_inches='tight')
    
    #Combining with trace plot
    list_im = ['vis/' + im for im in ["Header"+str(i)+".png", 'trace'+str(i)+'.png']]
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    new_im = PIL.Image.new(imgs[0].mode, (imgs[1].size[0], imgs[0].size[1]))
    new_im.paste(imgs[0])

    imgs_comb = np.vstack( [new_im, imgs[1]] )
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    
    imgs_comb.save('vis/'+'vis'+str(i)+'.png' )
    
p = Pool()
p.map(OnePlot, range(len(DATA)))
