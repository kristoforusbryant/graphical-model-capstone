"""
Tests that the data generated corresponds to the graph
"""

import utils.Graph as Graph
from tests.test_utils.generate_data import generate_data
import numpy as np

def test_generate_data():
    for _ in range(10):
        n = 10
        m = 10000
        g = Graph(n)
        g.SetRandom()
        X = generate_data(n, m, g)

        a = g.GetAdjM()[np.triu_indices(n, 1)]
        b = (np.abs(np.linalg.inv((X.transpose() @ X) / (m-1))[np.triu_indices(n, 1)]) > 0.1).astype(int)

        threshold = n
        assert(np.sum(a != b) < threshold)