from .minimal_ordering import * 
import copy 
def graph_union(d0,d1): 
    for k, l in d1.items(): 
        d0[k] = list(set(d0[k]).union(set(d1[k])))
    return d0

def F(dol, orderings=None):
    if orderings is None: 
        orderings = LEXM(dol)
    F_dol = copy.deepcopy(dol)
    for v in orderings:
        for nb in F_dol[v]:
            F_dol[nb] = list(set(F_dol[nb]).union(set(F_dol[v])))
            F_dol[nb].remove(nb)
            F_dol[nb].remove(v)
        del F_dol[v]
        dol = graph_union(dol, F_dol)
    return graph_union(dol, F_dol)

def C(dol, v, ordering=None, F_dol=None):
    if ordering is None: 
        ordering = LEXM(dol)
    if F_dol is None: 
        F_dol = F(dol, ordering)
    return set([w for w in F_dol[v] if ordering.index(w) > ordering.index(v)])

def one_connected_comp(d, start):
    to_visit = [start]
    visited = set()
    while to_visit:
        temp = set()
        for v in to_visit: 
            visited = visited.union({v})
            temp = temp.union(set(d[v]))
        to_visit = list(temp - visited)
    return visited 

def subgraph(g, nodes):
    d = {}
    for n in list(nodes):
        d[n] = list(set(g[n]).intersection(set(nodes))) 
    return d

def isclique(d):
    for v in d.keys(): 
        if set(d[v]) != set(d.keys()) - set([v]): return False
    return True

# Algorithm Implemented from (Tarjan, 1983)
def primecomps(dol): 
    primes = []
    separators = []
    ordering = LEXM(dol)
    F_dol = F(copy.deepcopy(dol), ordering)
    C_list = [C(dol,v, ordering, F_dol) for v in dol.keys()]
    vert_l = ordering
    while(vert_l): 
        i = vert_l[0]
        vert_l = vert_l[1:]
        V = set(dol.keys())
        A = one_connected_comp(subgraph(dol, V - C_list[i]), i)
        B = V - (C_list[i].union(A))
        if isclique(subgraph(dol, C_list[i])) and B: 
            primes.append(A.union(C_list[i]))
            separators.append(C_list[i])
            dol = subgraph(dol, B.union(C_list[i]))
            for v in vert_l: # removing vertices in B cup C[v]
                if v not in dol.keys(): 
                    vert_l.remove(v)
    primes.append(A.union(C_list[i]))
    for p in primes: 
        if p in separators: 
            primes.remove(p) 
            separators.remove(p)
    return (primes, separators)