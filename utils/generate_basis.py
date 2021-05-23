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
    B = GF2(np.zeros((m, cb)).astype(int))
    idx = 0
    for i in range(len(T)):
        for j in range(i):
            if j in T._dol[i]: continue
            T_ = T.copy()
            T_.AddEdge(i,j)
            B[:, idx] = T_.GetSubgraph(T_.find_one_cycle(i, -1,[])).GetBinaryL()
            idx += 1
    return B

def cycle_basis_complete(T):
    m = len(T) * (len(T) - 1) // 2
    B = GF2(np.zeros((m, m)).astype(int))
    idx = 0
    for i in range(len(T)):
        for j in range(i):
            if j in T._dol[i]: continue
            T_ = T.copy()
            T_.AddEdge(i,j)
            B[:, idx] = T_.GetSubgraph(T_.find_one_cycle(i, -1,[])).GetBinaryL()
            idx += 1
    for i in np.where(T.GetBinaryL())[0]:
        B[i, idx] = 1
        idx += 1
    return B

def cut_basis(T):
    # Assuming Complete Graph
    B = np.zeros((len(T) - 1, len(T) - 1).astype(int))
    idx = 0
    for i,j in T.GetEdgeL():
        T_minus_e = T.copy()
        T_minus_e.RemoveEdge(i,j)
        g = Graph(len(T_minus_e))
        g.SetFromEdgeL(list(product(T_minus_e.ConnectedTo(i), T_minus_e.ConnectedTo(j))))
        B[:, idx] = g.GetBinaryL()
        idx += 1
    return B


# change of basis matrix M: current -> std
def COBM(T):
    basis = cycle_basis(T) + cut_basis(T)
    M = np.zeros((len(basis), len(basis)), dtype=int)
    for i in range(len(basis)):
        M[:,i] = basis[i].GetBinaryL()
    return M

