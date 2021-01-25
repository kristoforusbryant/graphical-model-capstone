import os, sys
import numpy as np
import pickle
import json
from utils.diagnostics import create_edge_matrix
import PIL
import matplotlib.pyplot as plt
from utils.Graph import Graph

# Get Input project dir 
exppath = str(sys.argv[1]) 
#exppath = "experiments/five_on_five"

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
    
    sampler.SaveRaw(os.path.join(exppath, 'res/raw_'+str(i)+'.pkl'))
    sampler.SaveSummary(os.path.join(exppath, 'summary/summary_'+str(i)+'.pkl'))

    # Define the statistics to be traced
    triu = np.triu_indices(n, 1) # upper-tri indices 
    triu_edge_list = [(triu[0][i], triu[1][i]) for i in range(len(triu[0]))]
    dof = {"(" + str(i) + "," + str(j) + ")" : lambda x,i=i,j=j: x.IsEdge(i, j) 
           for i,j in triu_edge_list}
    add_dicts = [{"posterior": np.array(sampler.res[rep]['LIK']) + 
                  np.array(sampler.res[rep]['PRIOR'])} for rep in range(reps)]
    
    # Plot Traces
    fig = sampler.GetTrace(dof, os.path.join(exppath, 'vis') + "/trace" + str(i) + '.png', 
                           additional_dicts = add_dicts, list_first=True, burnin=0) # CONFIG['BURNIN'][i]
    
    # Plot True and Preidcted Graphs
    AdjM_list = [create_edge_matrix(rep['SAMPLES']) for rep in sampler.res]
    AdjM = sum(AdjM_list) / len(AdjM_list)
    AdjM[AdjM >= .5] = 1
    AdjM[AdjM < .5] = 0
    
    g = Graph(n)
    g.SetFromAdjM(AdjM)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    PARAMS[i].Draw(ax = axes[0])
    axes[0].set_title('true', fontsize=20)
    g.Draw(ax = axes[1]) 
    axes[1].set_title('predicted', fontsize=20)
    fig.tight_layout()

    filename = exppath + '/vis/' + "Graphs"+str(i)+".png"
    fig.savefig(filename, dpi=250, bbox_inches='tight')
    
    list_im = [exppath + '/vis/' + im for im in ["Graphs"+str(i)+".png", 'trace'+str(i)+'.png']]
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    new_im = PIL.Image.new(imgs[0].mode, (imgs[1].size[0], imgs[0].size[1]))
    new_im.paste(imgs[0])

    imgs_comb = np.vstack( [new_im, imgs[1]] )
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    
    imgs_comb.save( exppath + '/vis/'+'vis'+str(i)+'.png' )