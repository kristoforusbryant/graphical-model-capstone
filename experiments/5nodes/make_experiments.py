from dists.Params import GraphAndBasis
from dists.CountPriors import TruncatedNB
from dists.Priors import BasisCount
from dists.TreePriors import Uniform
from utils.generate_basis import circle_graph
from tests.test_utils.generate_data import generate_data_and_save

import numpy as np
import networkx as nx
import pickle

# Number of nodes and observations
n, n_obs = 5, 250

# Empty Graph
empty = GraphAndBasis(n)

# Circle Graph
circle = GraphAndBasis(n, circle_graph(n))

# Random Graphs from the Cycle space
print("Generating Graphs from the Cycle space...")
ct_prior = TruncatedNB(10, .75)
prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))
np.random.seed(123)
random0 = prior.Sample()
random1 = prior.Sample()

# Random Graphs from the Cycle space complement
print("Generating Graphs from the Cycle space complement...")
np.random.seed(123)
random2 = GraphAndBasis(n, nx.to_dict_of_lists(nx.erdos_renyi_graph(n, .2, seed=123)))
random3 = GraphAndBasis(n, nx.to_dict_of_lists(nx.barabasi_albert_graph(n, 2, seed=123)))

# List of graphs
graphs = [empty, circle, random0, random1, random2, random3]
graph_names = ['empty', 'circle', 'random0', 'random1', 'random2', 'random3']

# Generate data
for i in range(len(graphs)):
    g = graphs[i]
    outfile = "data/" + graph_names[i] + '.dat'
    print('Generating ' + outfile + '...' )
    generate_data_and_save(n, n_obs, g, outfile, threshold=None, seed=123)

# Saving graphs
for i in range(len(graphs)):
    g = graphs[i]
    g.SetName(graph_names[i])
    outfile = "data/graph_" + graph_names[i] + '.pkl'
    print('Saving ' + outfile + '...' )
    with open(outfile, 'wb') as handle:
        pickle.dump(g, handle)
