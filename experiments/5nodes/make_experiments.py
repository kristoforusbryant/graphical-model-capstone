from dists.Params import GraphAndBasis as Param
from tests.test_utils.generate_data import generate_data_and_save
from utils.generate_basis import circle_graph
import numpy as np
import networkx as nx

# Number of nodes and observations
n, n_obs = 5, 0

# Graphs to test on
graphs = []

# Empty Graph
graphs.append(Param(n))

# Circle Graph
g = Param(n, circle_graph(n))
graphs.append(g.copy())

# Random Graph Cycle Space
np.random.seed(123)
from dists.TreePriors import Hubs as TreePrior
from dists.Priors import BasisInclusion as Prior
tree_prior = TreePrior(n)
prior = Prior(n, Param, alpha=.5, tree_prior=tree_prior)

g = Param(n)
n_edges = n * (n - 1) // 2
while g.EdgeCount() <= int(.1 * n_edges) or g.EdgeCount() > int(.4 * n_edges):
    g = prior.Sample()
graphs.append(g.copy())
g = Param(n)
while g.EdgeCount() <= int(.6 * n_edges) or g.EdgeCount() > int(.9 * n_edges):
    g = prior.Sample()
graphs.append(g.copy())

# Random Graph
### change this to only graph in the cycle space
g = Param(n, nx.to_dict_of_lists(nx.erdos_renyi_graph(n, .2, seed=123)))
graphs.append(g.copy())
g = Param(n, nx.to_dict_of_lists(nx.barabasi_albert_graph(n, 2, seed=123)))
graphs.append(g.copy())

names = ["empty", "circle", "random0", "random1", "random2", "random3"]
assert(len(names) == len(graphs))

for i in range(len(names)):
    generate_data_and_save(n, n_obs, graphs[i], f"data/{names[i]}_{n}_{n_obs}.dat", threshold=None, seed=123)

import pickle
for i in range(len(names)):
    with open(f"data/graph_{names[i]}_{n}_{n_obs}.pkl", 'wb') as handle:
        pickle.dump(graphs[i], handle)
