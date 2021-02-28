def complete_path(n):
    dol = {i:[i-1,i+1] for i in range(1,n-1)}
    dol[0] = [1]
    dol[n-1] = [n-2]
    return dol 

def hub(n):
    dol =  {i:[0] for i in range(1,n)}
    dol[0] = list(range(1, n))
    return dol

def hybrid(n, k):
    # path part
    dol = {i:[i-1,i+1] for i in range(1,k-1)}
    dol[0] = [1]
    # hub part
    for i in range(k+1, n):
        dol[i] = [k]
    dol[k] = [k-1] + list(range(k+1,n))
    return dol

def cycle_basis(T): 
    # TODO: check if T is a spanning tree of its nodes
    basis = []
    for i in range(len(T)): 
        for j in range(i):
            if j in T._dol[i]: continue
            T_ = T.copy()
            T_.AddEdge(i,j)
            basis.append(T_.GetSubgraph(T_.find_one_cycle(i, -1,[])))
    return basis