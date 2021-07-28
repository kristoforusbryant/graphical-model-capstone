from networkx.classes import graph
from dists.Params import GraphAndBasis
from dists.CountPriors import TruncatedNB
from dists.Priors import BasisCount, BasisInclusion
from dists.TreePriors import Uniform
from utils.generate_basis import circle_graph
from tests.test_utils.generate_data import generate_data_and_save

import numpy as np
import networkx as nx
import pickle

# Number of nodes and observations
n, n_obs = 100, 1000

# Empty Graph
empty = GraphAndBasis(n)

# Circle Graph
circle = GraphAndBasis(n, circle_graph(n))

# Random Graphs from the Cycle space
print("Generating Graphs from the Cycle space...")
ct_prior = TruncatedNB(1, .5)
prior = BasisCount(n, GraphAndBasis, ct_prior, Uniform(n))
np.random.seed(123)
random0 = prior.Sample()
random1 = prior.Sample()

# Random Graphs from the Cycle space complement
print("Generating Graphs from the Cycle space complement...")
random2 = GraphAndBasis(n, nx.to_dict_of_lists(nx.erdos_renyi_graph(n, .05, seed=123)))
random3 = GraphAndBasis(n, nx.to_dict_of_lists(nx.barabasi_albert_graph(n, 2, seed=123)))

prior = BasisInclusion(n, GraphAndBasis, alpha=.05, tree_prior=Uniform(n))
np.random.seed(123)
random4 = prior.Sample()
random5 = prior.Sample()

# List of graphs
graphs = [empty, circle, random0, random1, random2, random3, random4, random5]
names = ['empty', 'circle', 'random0', 'random1', 'random2', 'random3', 'random4', 'random5']

for i in range(len(names)):
    generate_data_and_save(n, n_obs, graphs[i], f"data/{names[i]}_{n}_{n_obs}.dat", threshold=None, seed=123)

import pickle
for i in range(len(names)):
    with open(f"data/graph_{names[i]}_{n}_{n_obs}.pkl", 'wb') as handle:
        pickle.dump(graphs[i], handle)
