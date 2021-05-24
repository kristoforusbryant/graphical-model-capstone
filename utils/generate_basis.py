from itertools import product
from utils.Graph import Graph
import numpy as np
import galois
GF2 = galois.GF(2)

def complete_path(n):
    dol = {i:[i-1,i+1] for i in range(1,n-1)}
    dol[0] = [1]
    dol[n-1] = [n-2]
    return dol

def hub(n, v=0):
    not_v = set(range(n)) - set([v])
    dol =  {i:[v] for i in not_v}
    dol[v] = list(not_v)
    return dol

def hybrid(n, k):
    # path part
    dol = {i:[i-1,i+1] for i in range(1,k)}
    dol[0] = [1]
    # hub part
    for i in range(k, n):
        dol[i] = [k]
    dol[k] = [k-1] + list(range(k,n))
    return dol

def circle_graph(n):
    dol = {i:[i-1,i+1] for i in range(1,n-1)}
    dol[0] = [1, n-1]
    dol[n-1] = [0, n-2]
    return dol

def edge_basis(n):
    m = n * (n - 1) // 2
    return GF2(np.eye(m).astype(int))


def cycle_basis(T):
    n = len(T)
    m = n * (n - 1) // 2
    cb = (n - 1) * (n - 2) // 2
    B = GF2(np.zeros((m, cb), dtype=int))
    idx = 0
    for i in range(len(T)):
        for j in range(i):
            if j in T._dol[i]: continue
            T.AddEdge(i,j)
            B[:, idx] = T.GetSubgraph(T.find_one_cycle(i, -1,[])).GetBinaryL()
            T.RemoveEdge(i,j)
            idx += 1
    return B

def cycle_basis_complete(T):
    m = len(T) * (len(T) - 1) // 2
    B = GF2(np.zeros((m, m), dtype=int))
    idx = 0
    for i in range(len(T)):
        for j in range(i):
            if j in T._dol[i]: continue
            T.AddEdge(i,j)
            B[:, idx] = T.GetSubgraph(T.find_one_cycle(i, -1,[])).GetBinaryL()
            T.RemoveEdge(i,j)
            idx += 1
    for i in np.where(T.GetBinaryL())[0]:
        B[i, idx] = 1
        idx += 1
    return B

def cut_basis(T):
    # Assuming Complete Graph
    B = np.zeros((len(T) - 1, len(T) - 1), dtype=int)
    idx = 0
    for i,j in T.GetEdgeL():
        T.RemoveEdge(i,j)
        g = Graph(len(T))
        g.SetFromEdgeL(list(product(T.ConnectedTo(i), T.ConnectedTo(j))))
        B[:, idx] = g.GetBinaryL()
        T.AddEdge(i,j)
        idx += 1
    return B

def change_basis(g, t_):
    """
    Change the basis of g to that by t_
    """
    # M @ b -> e
    M_ = GF2(cycle_basis_complete(t_))
    active_ = np.linalg.solve(M_, GF2(g.GetBinaryL()))

    n = len(g)
    cb = (n - 1) * (n - 2) // 2
    g._basis_active = active_[:cb]
    g._tree = t_
    g._basis = M_[:, :cb]

    return g
