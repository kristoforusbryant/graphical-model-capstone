import numpy as np 

def thin(a, lag=1):
    indices = [i for i in range(len(a)) if i % lag == 0]
    return a[indices]

def ACF(a):
    n = len(a)
    a_bar = np.mean(a) 
    a = a - a_bar # de-mean a 
    normalising = np.sum(a * a)
    
    acf = [ np.sum(a[i:] * a[:n - i]) for i in range(n)]
    return np.array(acf) / normalising     

def IAC_time(a, M = None):
    # Implementation according to https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    if not M: M = len(a)

    n = len(a)
    a_bar = np.mean(a) 
    a = a - a_bar # de-mean a 
    normalising = np.sum(a * a) / len(a)
    
    corr = np.array([ np.sum(a[i:] * a[:n - i]) / (n-i) for i in range(n)]) / normalising
    return 1 + 2 * np.sum(corr[:M])