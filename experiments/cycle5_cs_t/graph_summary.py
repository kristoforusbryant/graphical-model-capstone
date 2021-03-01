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
import matplotlib.pyplot as plt
from utils.Graph import Graph
from utils.diagnostics import create_edge_matrix_from_binary_str
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score 

# Reading Config file 
filename=os.path.join(filepath, "config.json")
with open(filename) as f: 
    CONFIG = json.load(f)

N_graphs = len(CONFIG['N_NODES'])
assert N_graphs == len(CONFIG['N_REPS']), "len(nodes) != len(reps)"
assert N_graphs == len(CONFIG['N_DATA']), "len(nodes) != len(data)"
assert N_graphs == len(CONFIG['ITER']), "len(nodes) != len(iter)"
assert N_graphs == len(CONFIG['BURNIN']), "len(nodes) != len(burnin)"

### Load Params and Summary  
filename = os.path.join(filepath, 'Params.pkl')
with open(filename, 'rb') as handle:
    PARAMS = pickle.load(handle)

for i in range(N_graphs): 
    filename_r = os.path.join(filepath, 'res/raw_' + str(i) + '.pkl')
    with open(filename_r, 'rb') as handle:
        res = pickle.load(handle)
    res = res[0]
    
    filename_s = os.path.join(filepath, 'summary/summary_' + str(i) + '.pkl')
    with open(filename_s, 'rb') as handle:
        summary = pickle.load(handle)
        
    for rep in range(CONFIG["N_REPS"][i]): 
        AdjM = create_edge_matrix_from_binary_str(res[rep]['SAMPLES'], len(PARAMS[i])) 
        AdjM_flat = AdjM[np.triu_indices(len(PARAMS[i]), 1)]
        summary[rep]['ACCURACY'] = accuracy_score(PARAMS[i].GetBinaryL(), AdjM_flat > .5)
        try:
            summary[rep]['AUC'] = roc_auc_score(PARAMS[i].GetBinaryL(), AdjM_flat)
        except ValueError:
            summary[rep]['AUC'] = np.nan
        summary[rep]['F1'] = f1_score(PARAMS[i].GetBinaryL(), AdjM_flat > .5)
    
    with open(filename_s, 'wb') as handle: 
        pickle.dump(summary, handle)