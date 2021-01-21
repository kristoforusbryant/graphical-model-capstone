from math import gamma
import numpy as np
from .prime_components import primecomps

def LogGamma_n(x, n):
    acc = (n * (n-1)) / 4 * np.log(np.pi)
    for i in range(n):
        acc += np.log(gamma(x - i/2))
    return acc

def IG_clique(delta, D):
    n = D.shape[0]
    return (delta + n - 1) * n / 2 * np.log(2)  +  LogGamma_n((delta + n - 1)/2, n) - (delta + n -1)/2 * np.log(np.linalg.det(D))

def is_complete_subset(dol, nodes):
    for v in list(nodes): 
        if nodes - set(dol[v] + [v]): 
            return False
    return True

def IG_decomposable(dol, delta, D): 
    primes, seps = primecomps(dol)
    if not sum([is_complete_subset(dol, p) for p in primes]) > 0: 
        return np.nan
    
    acc = 0
    for p in primes: 
        acc += IG_clique(delta, D[list(p),:][:, list(p)])
    for s in seps:
        if s: 
            acc -= IG_clique(delta, D[list(s),:][:, list(s)])
    return acc