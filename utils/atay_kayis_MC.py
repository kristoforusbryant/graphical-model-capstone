import numpy as np
from math import gamma

def to_AdjM(d):
    n = len(d)
    M = np.zeros((n, n))
    for k,l in d.items():
        for i in l:
            M[i,k] = 1
            M[k,i] = 1
    return M

def sample_one(A, delta, T):
    n = A.shape[0]
    v = lambda i: np.sum(A[i,:])
    t_ = lambda i,j: T[i,j] / T[j,j]
    Phi = np.zeros((n,n))
    Phi[(np.arange(n),np.arange(n))] = np.sqrt([np.random.chisquare(df=delta + v(i)) for i in range(A.shape[0])]) # diagonal
    Phi[np.nonzero(np.triu(A))] = np.random.normal(size=len(np.nonzero(np.triu(A))[0])) # free edges

    fixed_idx = np.nonzero(np.triu(A== 0, k=1))
    idx = [(fixed_idx[0][i], fixed_idx[1][i]) for i in range(len(fixed_idx[0]))]
    acc = 0
    for i,j in idx:
        if i == 0:
            Phi[i,j] = -np.sum([Phi[i,k] * t_(k,j) for k in range(i,j-1)])
            acc += -.5 * Phi[i,j] * Phi[i,j]
        else:
            acc_ij = 0
            for k in range(i,j):
                acc_ij += Phi[i,k] * t_(k,j)
            for r in range(i):
                acc_r0 = 0
                for l in range(r, i):
                    acc_r0 += Phi[r,l] * t_(l,i)
                acc_r1 = 0
                for l in range(r, j):
                    acc_r1 += Phi[r,l] * t_(l,j)
                acc_ij += (Phi[r,i] + acc_r0)/ Phi[i,i] * (Phi[r,j] + acc_r1)
            Phi[i,j] = -acc_ij
            acc += -.5 * acc_ij * acc_ij
    return np.exp(acc)


def sample_one_identity(A, delta):
    n = A.shape[0]
    v = lambda i: np.sum(A[i,:])
    Phi = np.zeros((n,n))
    Phi[(np.arange(n),np.arange(n))] = np.sqrt([np.random.chisquare(df=delta + v(i)) for i in range(A.shape[0])]) # diagonal
    Phi[np.nonzero(np.triu(A))] = np.random.normal(size=len(np.nonzero(np.triu(A))[0])) # free edges

    fixed_idx = np.nonzero(np.triu(A== 0, k=1))
    idx = [(fixed_idx[0][i], fixed_idx[1][i]) for i in range(len(fixed_idx[0]))]
    acc = 0
    for i,j in idx:
        if i > 0:
            acc_ij = 0
            for r in range(i):
                acc_ij += Phi[r,i] / Phi[i,i] * Phi[r,j]
            Phi[i,j] = -acc_ij
            acc += -.5 * acc_ij * acc_ij
    return np.exp(acc)


def Atay_Kayis_MC(dol, delta, D, it=1000):
    # Step 1
    A = np.triu(to_AdjM(dol))
    # Step 2
    v = lambda i: np.sum(A[i,:])
    k = lambda j: np.sum(A[:,j])
    b = lambda i: v(i) + k(i) + 1
    T = np.linalg.cholesky(np.linalg.inv(D)).transpose()
    # Step 3 - 5
    if (D == np.eye(D.shape[0])).all():
        J_MC = np.mean([sample_one_identity(A, delta) for _ in range(it)]) # simplifies evaluation alot
    else:
        J_MC = np.mean([sample_one(A, delta, T) for _ in range(it)])
    # Step 6
    C = np.sum([ v(i)/2*np.log(2*np.pi) +
             (delta + v(i))/2*np.log(2) +
             np.log(gamma((delta + v(i))/2)) +
             (delta + b(i) - 1)*np.log(T[i,i]) for i in range(len(dol))])
    return C + np.log(J_MC)