import numpy as np
from dists.TreePriors import Uniform, Hubs, Paths

# aux variables
def is_hub_degrees(t):
    deglist = [len(l) for _, l in t._dol.items()]
    x = np.sum(np.array(deglist) == 1)
    y = np.sum(np.array(deglist) == (len(t) - 1))
    return (x == (len(t) - 1)) and (y == 1)

def is_path_degrees(t):
    deglist = [len(l) for _, l in t._dol.items()]
    x = np.sum(np.array(deglist) == 1)
    y = np.sum(np.array(deglist) == 2)
    return (x == 2) and (y == (len(t) - 2))


def test_treeprior_Uniform_Sample():
    n = 20
    treeprior = Uniform(n)
    np.random.seed(123)
    for _ in range(50):
        t= treeprior.Sample()
        assert(t.EdgeCount() == (n - 1))
        assert(len(t) == n)

def test_treeprior_Hubs_Sample():
    n = 20
    treeprior = Hubs(n)
    np.random.seed(123)
    for _ in range(50):
        t= treeprior.Sample()

        assert(t.EdgeCount() == (n - 1))
        assert(len(t) == n)
        assert(is_hub_degrees(t))

def test_treeprior_Paths_Sample():
    n = 20
    treeprior = Paths(n)
    np.random.seed(123)
    for _ in range(50):
        t= treeprior.Sample()

        assert(t.EdgeCount() == (n - 1))
        assert(len(t) == n)
        assert(is_path_degrees(t))

