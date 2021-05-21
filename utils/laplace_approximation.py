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


# Speeding up Laplace Approximation using C++ code
import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
os.environ["EXTRA_CLING_ARGS"] = "-I/usr/local/include -DNDB"

import cppyy
import igraph

# We load igraph in the following fashion since it is a C instead of a C++ library.
cppyy.c_include("igraph/igraph.h")
cppyy.load_library("libigraph")

cppyy.cppdef("""
#include <cmath>
#include <iostream>
#include <vector>

// This code was tested using Blaze version 3.8.0
// (https://bitbucket.org/blaze-lib/blaze/src/master/) with the fix from this
// pull request: https://bitbucket.org/blaze-lib/blaze/pull-requests/46.
//#include </usr/local/include/blaze/math/DynamicMatrix.h>
//#include </usr/local/include/blaze/math/DynamicVector.h>
//#include </usr/local/include/blaze/math/Column.h>
//#include </usr/local/include/blaze/math/Columns.h>
//#include </usr/local/include/blaze/math/Row.h>
//#include </usr/local/include/blaze/math/Rows.h>
//#include </usr/local/include/blaze/math/Elements.h>
//#include </usr/local/include/blaze/math/Subvector.h>
//#include </usr/local/include/blaze/math/Band.h>
#include </usr/local/include/blaze/Math.h>

blaze::DynamicMatrix<double> inv_pos_def(blaze::DynamicMatrix<double> mat) {
    blaze::invert<blaze::byLLH>(mat);
    return mat;
}


template <class T>
double log_det(T& mat) {
    // Log of the matrix determinant of `mat`
    blaze::DynamicMatrix<double> L;
    llh(mat, L);
    return 2.0 * sum(log(diagonal(L)));
}


blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > gwish_mode_inv(
    igraph_t* G_ptr, double df,
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >& rate
) {
    /*
    Find the inverse of the mode of a G-Wishart distribution.

    `G_ptr` is a pointer the graph to which the precision is constraint.

    `df` is the degrees of freedom of the distribution.

    `rate` is the rate or inverse scale matrix of the distribution and must be
    symmetric positive definite.

    The notation in this function follows Section 2.4 in
    Lenkoski (2013, arXiv:1304.1350v1).

    The optimization procedure is presented in Algorithm 17.1 of
    the Elements of Statistical Learning by Hastie et al.

    Compare Equation 7 from
    https://proceedings.neurips.cc/paper/2009/hash/a1519de5b5d44b31a01de013b9b51a80-Abstract.html
    with Equation 17.11 of the Elements of Statistical Learning by
    Hastie et al. to understand why we set `Sigma = rate / (df - 2.0)`.
    */
    int p = rate.rows();
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
        Sigma(rate / (df - 2.0)), W(Sigma);  // Step 1

    // Inspired by C++ code from the R package BDgraph
    std::vector<std::vector<double> > neighbors(p);
    std::vector<blaze::DynamicVector<double> > Sigma_N(p);
    // Avoid recomputing the neighbors for each iteration:
    igraph_vector_ptr_t igraph_neighbors;
    igraph_vector_ptr_init(&igraph_neighbors, p);

    igraph_neighborhood(
        G_ptr, &igraph_neighbors, igraph_vss_all(), 1, IGRAPH_ALL, 1
    );

    for (int i = 0; i < p; i++) {
        igraph_vector_t* N_ptr
            = (igraph_vector_t*) igraph_vector_ptr_e(&igraph_neighbors, i);

        neighbors[i].resize(igraph_vector_size(N_ptr));

        for (int j = 0; j < neighbors[i].size(); j++)
            neighbors[i][j] = igraph_vector_e(N_ptr, j);

        Sigma_N[i] = elements(column(Sigma, i), neighbors[i]);
    }

    IGRAPH_VECTOR_PTR_SET_ITEM_DESTRUCTOR(
        &igraph_neighbors, igraph_vector_destroy
    );

    igraph_vector_ptr_destroy_all(&igraph_neighbors);
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > W_previous(p);
    blaze::DynamicMatrix<double, blaze::columnMajor> W_N;
    blaze::DynamicMatrix<double, blaze::rowMajor> W_NN;
    blaze::DynamicVector<double> W_beta_hat(p), beta_star;

    for (int i = 0; i < 10000; i++) {
        W_previous = W;

        for (int j = 0; j < p; j++) {
            if (neighbors[j].size() == 0) {
                // This only happens if the graph `G_ptr` is not connected.
                W_beta_hat = 0.0;
            } else if (neighbors[j].size() == p - 1) {
                subvector(W_beta_hat, 0, j) = subvector(Sigma_N[j], 0, j);

                subvector(W_beta_hat, j + 1, p - j - 1)
                    = subvector(Sigma_N[j], j, p - j - 1);
            } else {
                W_N = columns(W, neighbors[j]);
                W_NN = rows(W_N, neighbors[j]);
                solve(declsym(W_NN), beta_star, Sigma_N[j]);
                W_beta_hat = W_N * beta_star;
            }

            double W_jj = W(j, j);
            column(W, j) = W_beta_hat;
            // The next line is not need as Blaze enforces the symmetry of `W`.
            // row(W, j) = trans(W_beta_hat);
            W(j, j) = W_jj;
        }

        // 1e-8 is consistent with BDgraph.
        if (blaze::mean(blaze::abs(W - W_previous)) < 1e-8) return W;
    }

    std::cout << "`gwish_mode_inv` failed to converge." << std::endl;
    return W;
}


blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > hessian(
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >& K_inv,
    igraph_t G_V, double df
) {
    // Compute the Hessian divided by `-0.5 * (df - 2.0)`.
    int n_e = igraph_ecount(&G_V);
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> > H(n_e);

    for (int a = 0; a < n_e; a++) {
        int i = IGRAPH_FROM(&G_V, a), j = IGRAPH_TO(&G_V, a);

        for (int b = a; b < n_e; b++) {
            int l = IGRAPH_FROM(&G_V, b), m = IGRAPH_TO(&G_V, b);

            if (i == j) {
                if (l == m) {
                    H(a, b) = std::pow(K_inv(i, l), 2);
                } else {
                    H(a, b) = 2.0 * K_inv(i, l) * K_inv(i, m);
                }
            } else {
                if (l == m) {
                    H(a, b) = 2.0 * K_inv(i, l) * K_inv(j, l);
                } else {
                    H(a, b) = 2.0 * (
                        K_inv(i, l)*K_inv(j, m) + K_inv(i, m)*K_inv(j, l)
                    );
                }
            }
        }
    }

    return H;
}


double log_gwish_norm_laplace(
    igraph_t* G_ptr, double df,
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >& rate,
    bool diag = false
) {
    /*
    Log of Laplace approximation of G-Wishart normalization constant

    Log of the Laplace approximation of the normalization constant of the
    G-Wishart distribution outlined by
    Lenkoski and Dobra (2011, doi:10.1198/jcgs.2010.08181)

    `diag` indicates whether the full Hessian matrix should be used
    (`diag = false`) or only its diagonal (`diag = true`). The latter is faster
    but less accurate per Moghaddam et al. (2009,
    https://papers.nips.cc/paper/2009/hash/a1519de5b5d44b31a01de013b9b51a80-Abstract.html).
    */
    int p = rate.rows(), n_e = igraph_ecount(G_ptr);

    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
        K_inv = gwish_mode_inv(G_ptr, df, rate);

    blaze::DynamicMatrix<double> K = inv_pos_def(K_inv);
    double log_det_H, h = -0.5 * (trace(K * rate) - (df - 2.0)*log_det(K));

    if (diag) {
        log_det_H = 2.0*sum(log(diagonal(K_inv))) + n_e*std::log(2.0);

        for (int vid = 0; vid < n_e; vid++) {
            int i = IGRAPH_FROM(G_ptr, vid), j = IGRAPH_TO(G_ptr, vid);

            log_det_H +=
                std::log(K_inv(i, i)*K_inv(j, j) + std::pow(K_inv(i, j), 2));
        }
    } else {
        // Create graph `G_V` which equals `G` plus all self-loops such that
        // its edge set coincides with Equation 2.1 of
        // Lenkoski and Dobra (2011).
        igraph_t G_V;
        igraph_copy(&G_V, G_ptr);
        igraph_vector_t edge_list;
        igraph_vector_init(&edge_list, 2 * p);

        for (int i = 0; i < p; i++) {
            VECTOR(edge_list)[2*i] = i;
            VECTOR(edge_list)[2*i + 1] = i;
        }

        igraph_add_edges(&G_V, &edge_list, 0);
        igraph_vector_destroy(&edge_list);

        blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
            H = hessian(K_inv, G_V, df);

        igraph_destroy(&G_V);
        log_det_H = log_det(H);
    }

    // The sign of the Hessian `-0.5 * (df - 2.0) * H` is flipped compared to
    // Lenkoski and Dobra (2011, Section 4). I think that this is correct as
    // |Hessian| can be negative while |-Hessian| cannot.
    return h + 0.5*(p + n_e)*std::log(2.0 * M_PI)
        - 0.5*((p + n_e)*std::log(0.5 * (df - 2.0)) + log_det_H);
}


double log_gwish_norm_laplace_cpp(
    igraph_t* G_ptr, double df, double* rate_in, bool diag = false
) {
    blaze::SymmetricMatrix<blaze::DynamicMatrix<double> >
        rate(igraph_vcount(G_ptr), rate_in);

    return log_gwish_norm_laplace(G_ptr, df, rate, diag);
}
""")

# Reduce Python overhead:
# If the next line fails, then the compilations of the C++ code might have failed.
log_gwish_norm_laplace_cpp = cppyy.gbl.log_gwish_norm_laplace_cpp

def laplace_approx (G, delta, D, as_log_prob=True, diag=False):
    """
    Log of Laplace approximation of G-Wishart normalization constant

    Log of the Laplace approximation of the normalization constant of the G-Wishart
    distribution outlined by Lenkoski and Dobra (2011, doi:10.1198/jcgs.2010.08181)
    """
    from utils.Graph import Graph
    G = Graph(len(G), G)
    G = igraph.Graph.Adjacency(G.GetAdjM().tolist(), mode=1)
    df = delta
    rate = D

    if rate is None:
        # If the rate matrix is the identity matrix (I_p), then the mode of the
        # G-Wishart distribution is (df - 2) * I_p such that the Laplace approximation simplifies.
        p = G.vcount()
        n_e = G.ecount()

        return 0.5 * (
            (p*(df - 1.0) + n_e)*np.log(df - 2.0) - p*(df - 2.0) \
                + p*np.log(4.0 * np.pi) + n_e*np.log(2.0 * np.pi)
        )

    return log_gwish_norm_laplace_cpp(G.__graph_as_capsule(), df, rate, diag)

