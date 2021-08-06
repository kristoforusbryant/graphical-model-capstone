"""
Given a graph, simulate data coming from one precision matrix corresponding to that particular graph.
"""

from utils.laplace_approximation import constrained_cov
import numpy as np
def generate_data(n, m, g, seed=None, threshold=.5, df=3):
    if seed is not None:
        np.random.seed(seed)

    T = np.random.random((df + n, n))
    C = T.transpose() @ T
    C_star = constrained_cov(g.GetDOL(), C, np.eye(n)) # constrain zeroes of the matrices
    K = np.linalg.inv(C_star)
    triu = np.triu_indices(n, 1)

    if threshold:
        count = 0
        while not ((np.abs(K[triu]) > threshold).astype(int) == g.GetBinaryL()).all():
            T = np.random.random((df + n, n))
            C = T.transpose() @ T
            C_star = constrained_cov(g.GetDOL(), C, np.eye(n)) # constrain zeroes of the matrices
            K = np.linalg.inv(C_star)
            triu = np.triu_indices(n, 1)

            if count > 50:
                raise ValueError("Can't find precision matrix with large enough non-zero values, \
                                try tweaking the threshold parameter.")

    assert(((np.abs(np.linalg.inv(C_star)) > 1e-10)[triu] == g.GetBinaryL()).all()) # zeros at the right places
    assert(np.allclose(C_star, C_star.transpose())) # symmetric
    assert(np.linalg.det(C_star) > 0.) # positive definite

    data = np.random.multivariate_normal(np.zeros(n), C_star, m)
    if m == 1:
        data = data.reshape(1, n)
    return data

def generate_data_and_save(n, m, g, outfile, **kwargs):
    data = generate_data(n, m, g, **kwargs)

    np.savetxt(outfile, data, delimiter=",")
    return 0

