import numpy as np
import copy
import random  
    
def BronKerbosch1(G, P, R=None, X=None):
    P = set(P)
    R = set() if R is None else R
    X = set() if X is None else X
    if not P and not X:
        yield R
    while P:
        v = P.pop()
        yield from BronKerbosch1(G,
            P=P.intersection(G[v]), R=R.union([v]), X=X.intersection(G[v]))
        X.add(v)

def BronKerbosch2(G, P, R=None, X=None):
    """ 
    Bron-Kerbosch Algorithm with Pivot from Bron and Kerbosch(1973)
    G: Graph as dict of lists
    """
    P = set(P)
    R = set() if R is None else R
    X = set() if X is None else X
    if not P and not X:
        yield R
    try:
        u = list(P.union(X))[0]
        S = P.difference(G[u])
    # if union of P and X is empty
    except IndexError:
        S = P
    for v in S:
        yield from BronKerbosch2(G, 
            P=P.intersection(G[v]), R=R.union([v]), X=X.intersection(G[v]))
        P.remove(v)
        X.add(v)
    
def mode(G, delta, D, N=100): 
    """
    Find the mode of a G-Wishart distribution.
    
    `G` (`Graph` object).
    `delta` is the degree of freedom of the distribution.
    `D` is the rate or inverse scale matrix of the distribution and must be symmetric positive definite.
    
    The notation in this function follows Section 2.4 in Lenkoski (2013, arXiv:1304.1350v1).
    
    The optimization procedure is presented in Algorithm 17.1 of
    the Elements of Statistical Learning by Hastie et al.
    
    Compare Equation 7 from
    https://proceedings.neurips.cc/paper/2009/hash/a1519de5b5d44b31a01de013b9b51a80-Abstract.html
    with Equation 17.11 of the Elements of Statistical Learning by Hastie et al.
    to understand why we set `Σ = rate / (df - 2.0)`.
    """
    n = len(G)
    L = D / (delta - 2)
    K = copy.deepcopy(L)
    index_rgwish = [np.delete(range(n), j) for j in range(n)]
    
    while True:
        K_ = copy.deepcopy(K)
        
        for j in range(n):
            β_hat_j = np.zeros(n)
            N_j = G[j]
            
            # Step 2a and 2b
            if len(N_j) > 0:
                β_hat_j[N_j] = np.linalg.solve(K[N_j, :][:, N_j], L[N_j, j])
            
            # Step 2c
            tmp = K[index_rgwish[j], :][:, index_rgwish[j]] @ β_hat_j[index_rgwish[j]]
            K[j, index_rgwish[j]] = tmp
            K[index_rgwish[j], j] = tmp
            
        # Step 3: Stop if converged.
        if np.mean((K - K_)**2) < 1e-8:
            break
    
    return np.linalg.inv(K) # Step 4

def constrained_cov(G, L, M, N=100):
    """
    The second cyclic algorithm in (Speed and Kiiveri, 1986)
    """
    C_l = list(BronKerbosch1(G, G.keys()))
    K = np.linalg.inv(M)
    K_ = K
    for i in range(N):
        for c in C_l:
            c = tuple(c)
            Q_inv = np.linalg.inv(K_) 
            Q_inv[np.ix_(c, c)] += np.linalg.inv(L[c,:][:,c]) - np.linalg.inv(K_[c,:][:,c]) # subset first, then take inv     
            K_ = np.linalg.inv(Q_inv)
        if np.max(np.abs(K - K_)) < 1e-100: break
        K = K_
    return K 

def hessian(K, G_V, delta):
    n_e = len(G_V)
    H = np.zeros(2 * (n_e,))
    K_inv = np.linalg.inv(K)
    
    for a in range(n_e):
        i, j = G_V[a]
        
        for b in range(a, n_e):
            l, m = G_V[b]
            
            if i == j:
                if l == m:
                    H[a, b] = K_inv[i, l]**2
                else:
                    H[a, b] = 2.0 * K_inv[i, l] * K_inv[i, m]
            else:
                if l == m:
                    H[a, b] = 2.0 * K_inv[i, l] * K_inv[j, l]
                else:
                    H[a, b] = 2.0 * (K_inv[i, l]*K_inv[j, m] + K_inv[i, m]*K_inv[j, l])
    
    # So far, we have only determined the upper triangle of H.
    H += H.T - np.diag(np.diag(H))
    return -0.5 * (delta - 2.0) * H

def laplace_approx (G, delta, D, as_log_prob=True): 
    """
    Log of Laplace approximation of G-Wishart normalization constant
    
    Log of the Laplace approximation of the normalization constant of the G-Wishart
    distribution outlined by Lenkoski and Dobra (2011, doi:10.1198/jcgs.2010.08181)
    """
    p = len(G)
    
    K = LA.mode(G, delta, D)
    V = []
    # creating duplication matrix 
    for k,l in G.items():
        V.append((k,k))
        for v in l: 
            if k < v: V.append((k,v))
                
    h = -0.5*(np.trace(np.transpose(K) @ D) - (delta - 2.0)*np.linalg.slogdet(K)[1])
    H = hessian(K, V, delta)
    print(H)
    print(V)
    print(len(V))
    print(np.linalg.slogdet(-H)[1])
    # The minus sign in front of `H` is not there in Lenkoski and Dobra (2011, Section 4).
    # I think that it should be there as |H| can be negative while |-H| cannot.
    if as_log_prob: 
        return h + 0.5*len(V)*np.log(2.0 * np.pi) - 0.5*np.linalg.slogdet(-H)[1]
    else: 
        log_p = h + 0.5*len(V)*np.log(2.0 * np.pi) - 0.5*np.linalg.slogdet(-H)[1]
        return np.exp(maxmin(log_p))
