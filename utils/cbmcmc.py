""" MCMC Inference of Gaussian Graphical Models using Cycle Basis Prior

This script allows users to obtain MCMC samples from of a Gaussian Graphical Model efficiently
by using the cycle basis prior.

This script requires the packages `numpy`, and `pandas` to be
installed within the Python environment you are running this script in.

Listed here are the required arguments:
    --data, path to file containing the observation matrix.
    --outfile, path to file that will contain the result.
    --it, number of MCMC iterations.

Listed here are the optional arguments:
    --basis, 'edge' or 'cycle' basis.
    --treeprior, spanning tree that generates the cycle basis ('all', 'hub', or 'path').
    -r, success parameter of the negative binomial hyperprior.
    -p, probability parameter of the negative binomial hyperprior.
    --cob-freq, frequency of change of basis.
    --seed, random number seed set at the start of the MCMC.

Author: Kristoforus Bryant Odang
"""

import sys
import numpy as np
from utils import Parser
from utils.generate_basis import edge_basis, cycle_basis
from utils.MCMC import MCMC_Sampler
import dists.TreePriors, dists.CountPriors
import dists.Priors, dists.Likelihoods, dists.Proposals
from dists.Params import GraphAndBasis

def cbmcmc(data, it=1000, basis='cycle', treeprior='all', r=None, p=.75, cob_freq=100, outfile=None, seed=123, init=None):
    data = np.loadtxt(data, delimiter=",")
    _, n = data.shape

    # Bases and Tree Prior
    if basis =='edge':
        basis = edge_basis(n)
        tree_prior = None
        cob_freq = None
    elif  basis == 'cycle':
        basis = None
        if  treeprior == 'all':
            tree_prior = dists.TreePriors.Uniform(n)
        elif  treeprior == 'hub':
            tree_prior = dists.TreePriors.Hubs(n)
        elif  treeprior == 'path':
            tree_prior = dists.TreePriors.Paths(n)
        else:
            raise ValueError("Invalid tree prior")
    else:
        raise ValueError("Invalid basis")

    # Count Prior
    if r is None:
        ct_prior = dists.CountPriors.TruncatedNB(data.shape[1], p)
    else:
        ct_prior = dists.CountPriors.TruncatedNB(r, p)

    # Prior, Likelihood, and Proposal
    prior = dists.Priors.BasisCount(n, GraphAndBasis, ct_prior, tree_prior, basis)
    lik = dists.Likelihoods.GW_LA(data, 3, np.eye(n), GraphAndBasis)
    prop = dists.Proposals.BasisWalk(n, GraphAndBasis, tree_prior, cob_freq)

    # Run MCMC
    print("Starting MCMC...")
    sampler = MCMC_Sampler(prior, prop, lik, data, outfile=outfile)
    if seed:
        np.random.seed(seed)

    if init is None:
        sampler.run(it)
    else:
        import pickle
        with open(init, 'rb') as handle:
            g = pickle.load(handle)
        sampler.run(it, fixed_init=handle)

    # Saving Results
    if outfile is not None:
        sampler.save_object()
    return sampler


def main():
    # Parsing user-given inputs
    parser = Parser.Parser(sys.argv[1:]).args

    cbmcmc(parser.data, parser.it, parser.basis, parser.treeprior,
           parser.r, parser.p, parser.cob_freq, parser.outfile, parser.seed)

    return 0

if __name__ == "__main__":
    main()