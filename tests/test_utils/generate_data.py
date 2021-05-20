"""
Given a graph, simulate data coming from one precision matrix corresponding to that particular graph.
"""

from utils.laplace_approximation import constrained_cov
import numpy as np
def generate_data(n, m, g, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T = np.random.random((n,n))
    C = T.transpose() @ T
    C_star = constrained_cov(g.GetDOL(), C, np.eye(n)) # constrain zeroes of the matrices

    data = np.random.multivariate_normal(np.zeros(n), C_star, m)
    return data

def generate_data_and_save(n, m, g, outfile, seed=None):
    data = generate_data(n, m, g, seed)

    np.savetxt(outfile, data, delimiter=",")
    return 0

