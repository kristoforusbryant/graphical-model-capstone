""" MCMC Inference of Gaussian Graphical Models using Cycle Basis Prior

This script allows users to obtain MCMC samples from of a Gaussian Graphical Model efficiently
by using the cycle basis prior.

The script should be run as:
`python cbmcmc.py --data observations.dat --outfile samples.pkl --it 10000`

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

def cbmcmc(data, outfile, it, basis='cycle', treeprior='all', r=6, p=.75, cob_freq=100, seed=123):
    data = np.loadtxt(data, delimiter=",")
    _, n = data.shape
    print(n)

    # Bases and Tree Prior
    if basis =='edge':
        basis = edge_basis(n)
        tree_prior = None
        cob_freq = None
    elif  basis == 'cycle':
        if  treeprior == 'all':
            tree_prior = dists.TreePriors.Uniform(n)
            basis = cycle_basis(tree_prior.Sample())
        elif  treeprior == 'hub':
            tree_prior = dists.TreePriors.Hubs(n)
            basis = cycle_basis(tree_prior.Sample())
        elif  treeprior == 'path':
            tree_prior = dists.TreePriors.Paths(n)
            basis = cycle_basis(tree_prior.Sample())
        else:
            raise ValueError("Invalid tree prior")
    else:
        raise ValueError("Invalid basis")

    # Count Prior
    ct_prior = dists.CountPriors.TruncatedNB(r, p)

    # Prior, Likelihood, and Proposal
    prior = dists.Priors.BasisCount(n, GraphAndBasis, ct_prior, tree_prior, basis)
    lik = dists.Likelihoods.GW_LA(data, 3, np.eye(n), GraphAndBasis)
    prop = dists.Proposals.BasisWalk(n, GraphAndBasis, tree_prior, cob_freq)

    # Run MCMC
    sampler = MCMC_Sampler(prior, prop, lik, data, outfile= outfile)
    if seed:
        np.random.seed(seed)
    sampler.run( it)

    # Saving Results
    sampler.save_object()
    return 0


def main():
    # Parsing user-given inputs
    parser = Parser.Parser(sys.argv[1:]).args

    cbmcmc(parser.data, parser.outfile, parser.it, parser.basis, parser.treeprior,
           parser.r, parser.p, parser.cob_freq, parser.seed)

    return 0

if __name__ == "__main__":
    main()