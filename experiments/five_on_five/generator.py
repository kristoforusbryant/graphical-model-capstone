import sys, os
from os.path import dirname
filepath = dirname(os.path.realpath(__file__)) # adding file directory 
sys.path.append(filepath)
projectpath = dirname(dirname(filepath)) # adding directory containing utils
sys.path.append(projectpath)

import numpy as np
import pickle
import json
from utils.laplace_approximation import constrained_cov

# Reading Config file 
filename=os.path.join(filepath, "config.json")
with open(filename) as f: 
    CONFIG = json.load(f)

N_graphs = len(CONFIG['N_NODES'])
assert N_graphs == len(CONFIG['N_REPS']), "len(nodes) != len(reps)"
assert N_graphs == len(CONFIG['N_DATA']), "len(nodes) != len(data)"
assert N_graphs == len(CONFIG['ITER']), "len(nodes) != len(iter)"
assert N_graphs == len(CONFIG['BURNIN']), "len(nodes) != len(burnin)"

### Generate the Graphs
from utils.Graph import Graph
n = CONFIG['N_NODES'][0] # graphs are manually design here

import importlib
Prior = importlib.import_module("dists.priors.uniform").Prior
prior = Prior(n) 

# Empty Graph 
empty = Graph(n)

# Circle Graph 
def circle_graph(n):
    dol = {i:[i-1,i+1] for i in range(1,n-1)}
    dol[0] = [1, n-1]
    dol[n-1] = [0, n-2]
    return dol 
circle = Graph(n, circle_graph(n))

# Random Graphs on 5 nodes 
np.random.seed(CONFIG['SEED'])
random_l = [prior.Sample() for _ in range(N_graphs - 2)]

# Joined 
GRAPHS = [empty, circle] + random_l 
    
    
### Generate data for each graph
K_MATRICES = []
DATA = []
for idx in range(N_graphs): 
    p = GRAPHS[idx]
    
    np.random.seed(idx*CONFIG['SEED'])
    T = np.random.random((n,n))
    C = T.transpose() @ T 
    C_star = constrained_cov(p.GetDOL(), C, np.eye(n)) # constrain zeroes of the matrices 
    K = np.round(np.linalg.inv(C_star), 10)
    
    np.random.seed(idx * CONFIG['SEED'])
    data = np.random.multivariate_normal(np.zeros(n), C_star, CONFIG['N_DATA'][idx])  
    # CONFIG['N_DATA'] observations,  n dimensions
    
    K_MATRICES.append(K.copy())
    DATA.append(data.copy())

# Saving     
filename = os.path.join(filepath, 'Params.pkl')
with open(filename, 'wb') as handle:
    pickle.dump(GRAPHS, handle)
    
filename = os.path.join(filepath, 'Data.pkl')
with open(filename, 'wb') as handle:
    pickle.dump(DATA, handle)
    
filename = os.path.join(filepath, 'K_matrices.pkl')
with open(filename, 'wb') as handle:
    pickle.dump(K_MATRICES, handle)