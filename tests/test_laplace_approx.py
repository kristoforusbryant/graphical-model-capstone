"""
Tests for Laplace Approximation of the G-Wishart normalising constant
"""

from utils.laplace_approximation import mode, constrained_cov, laplace_approx
from utils.Graph import Graph
import numpy as np

def test_constrained_cov_0():
    n = 10
    for _ in range(10):
        g = Graph(n)
        g.SetRandom()
        a = g.GetAdjM()[np.triu_indices(n, 1)]

        T = np.random.random((n,n))
        C = T.transpose() @ T
        C_star = constrained_cov(g.GetDOL(), C, np.eye(n))
        K = np.linalg.inv(C_star)
        b = (abs(K[np.triu_indices(n, 1)]) > 1e-10).astype(int)

        assert((a == b).all())

def test_constrained_cov_1():
    n = 30
    for _ in range(5):
        g = Graph(n)
        g.SetRandom()
        a = g.GetAdjM()[np.triu_indices(n, 1)]

        T = np.random.random((n,n))
        C = T.transpose() @ T
        C_star = constrained_cov(g.GetDOL(), C, np.eye(n))
        K = np.linalg.inv(C_star)
        b = (abs(K[np.triu_indices(n, 1)]) > 1e-10).astype(int)

        assert((a == b).all())


"""
Use the Iris-Virginica dataset and the empirical result by (Atay-Kayis & Massam 2005)
https://doi.org/10.1093/biomet/92.2.317 to test implementation of Laplace Approximation
"""

# Load Iris-Virginica dataset
from sklearn import datasets
iris = datasets.load_iris()

data = iris["data"][(
    iris["target"] == (iris["target_names"] == "virginica").nonzero()[0][0]
).nonzero()[0], :]
data -= data.mean(axis=0)

from utils.laplace_approximation import constrained_cov
import pandas as pd

n = 4
delta = 3
D = np.eye(4)
U = data.transpose() @ data

res = {
    'IG_EX_prior': [],
    'IG_MC_prior': [],
    'IG_LA_prior': [],
    'IG_CO_prior': [],
    'IG_EX_postr': [],
    'IG_MC_postr': [],
    'IG_LA_postr': [],
    'IG_CO_postr': [],
}

from tests.test_utils.get_all_graphs import get_all_graphs
from utils.G_Wishart import G_Wishart
all_graphs = get_all_graphs(n)
from tqdm import tqdm
for g in tqdm(all_graphs):
    D_star = constrained_cov(g.GetDOL(), D + U, np.eye(4))
    GW_prior = G_Wishart(g, delta, D)
    GW_postr = G_Wishart(g, delta + data.shape[0], D_star)
    res['IG_EX_prior'].append(GW_prior.IG_Exact())
    res['IG_MC_prior'].append(GW_prior.IG_MC(it=1000))
    res['IG_LA_prior'].append(GW_prior.IG_LA())
    res['IG_CO_prior'].append(GW_prior.IG())
    res['IG_EX_postr'].append(GW_postr.IG_Exact())
    res['IG_MC_postr'].append(GW_postr.IG_MC(it=1000))
    res['IG_LA_postr'].append(GW_postr.IG_LA())
    res['IG_CO_postr'].append(GW_postr.IG())

df = pd.DataFrame(res)
df['lik_EX'] = np.exp(df['IG_EX_postr'] - df['IG_EX_prior'])
df['lik_MC'] = np.exp(df['IG_MC_postr'] - df['IG_MC_prior'])
df['lik_LA'] = np.exp(df['IG_LA_postr'] - df['IG_LA_prior'])
df['lik_CO'] = np.exp(df['IG_CO_postr'] - df['IG_CO_prior'])
df['lik_EX'] = df['lik_EX'] / np.sum(df['lik_EX'])
df['lik_MC'] = df['lik_MC'] / np.sum(df['lik_MC'])
df['lik_LA'] = df['lik_LA'] / np.sum(df['lik_LA'])
df['lik_CO'] = df['lik_CO'] / np.sum(df['lik_CO'])

"""
Tests for Monte Carlo Approximation of the G-Wishart normalising constant
against empirical result presented in (Atay-Kayis & Massam 2005)
"""

def test_MC_approx():
    atay_kayis_values = np.array([.8212, 1, .4049, .5006, .9874, .5322])
    df['graphs'] = all_graphs
    prob = {tup:0 for tup in all_graphs[63].GetEdgeL()}
    for i in range(len(all_graphs)):
        for tup in all_graphs[i].GetEdgeL():
            prob[tup] += df['lik_MC'][i]

    assert(np.max(np.abs(atay_kayis_values - list(prob.values()))) < .01)

"""
Show that Laplace Approximation is close to MC and Exact.
"""

def test_laplace_approx_0():
    assert(np.max(np.abs(df['IG_EX_postr'] - df['IG_MC_postr']) < .5))
    assert(np.max(np.abs(df['IG_LA_postr'] - df['IG_MC_postr']) < .5))



