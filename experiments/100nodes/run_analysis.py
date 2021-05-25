import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.diagnostics import IAC_time, str_list_to_median_graph

def get_accuracies(g, md):
    l1= np.array(g.GetBinaryL(), dtype=bool)
    triu = np.triu_indices(len(g), 1)
    l2 = np.array(md[triu], dtype=bool)

    TP = np.logical_and(l1, l2).astype(int).sum()
    TN = np.logical_and(np.logical_not(l1), np.logical_not(l2)).astype(int).sum()
    FP = np.logical_and(np.logical_not(l1), l2).astype(int).sum()
    FN = np.logical_and(l1, np.logical_not(l2)).astype(int).sum()

    assert(TP + TN + FP + FN == len(l1))
    assert(TP + FP == l2.astype(int).sum())
    assert(TN + FN == np.logical_not(l2).astype(int).sum())

    return TP, TN, FP, FN

def plot_traces(a, title, outfile):
    fig = plt.figure(figsize=(20,10))
    plt.plot(a)
    plt.title(title, fontsize=30)
    plt.xlabel('Number of Iterations', fontsize=20)
    fig.savefig(outfile)
    plt.close(fig)
    return 0

def str_to_int_list(s):
    return np.array(list(s), dtype=int)

def main():
    files = [s for s in os.listdir('results') if '.pkl' in s]
    l = ['basis', 'graph', 'n', 'iter', 'time',
         'accept_rate', 'max_posterior', 'states_visited',
         'IAT_posterior', 'IAT_sizes', 'IAT_bases',
         'TP', 'TN', 'FP', 'FN']
    d = {k:[] for k in l}

    for f in files:
        with open('results/' + f, 'rb') as handle:
            sampler = pickle.load(handle)

        # Information about the experiments
        b_str, g_str = f.split('_')[:2]
        n = int(os.getcwd().split('/')[-1].replace('nodes', ''))
        d['basis'].append(b_str)
        d['graph'].append(g_str)
        d['n'].append(n)
        d['iter'].append(sampler.iter)
        d['time'].append(sampler.time)


        # Mixing Performance
        burnin = 15000
        posts = np.array(sampler.res['LIK'], dtype=float)[burnin:] + np.array(sampler.res['PRIOR'], dtype=float)[burnin:]
        sizes = list(map(lambda s: np.sum(str_to_int_list(s)), sampler.res['SAMPLES']))[burnin:]
        n_bases = list(map(lambda s: np.sum(str_to_int_list(sampler.lookup[s]['BASIS_ID'])), sampler.res['SAMPLES']))[burnin:]
        plot_traces(posts, 'Log Posterior', 'results/vis/post_traces/' + b_str + '_' + g_str + '_post_trace.pdf')
        plot_traces(sizes, 'Number of Edges', 'results/vis/size_traces/' + b_str + '_' + g_str + '_size_trace.pdf')
        plot_traces(n_bases, 'Number of Bases', 'results/vis/basis_traces/' + b_str + '_' + g_str + '_basisct_trace.pdf')

        d['IAT_posterior'].append(IAC_time(posts))
        d['IAT_sizes'].append(IAC_time(sizes))
        d['IAT_bases'].append(IAC_time(n_bases))

        d['accept_rate'].append(np.sum(sampler.res['ACCEPT_INDEX']) / len(sampler.res['ACCEPT_INDEX']))
        d['max_posterior'].append(np.max(posts))
        d['states_visited'].append(len(np.unique(sampler.res['SAMPLES'][burnin:])))


        # Accuracy Performance
        infile = 'data/graph_' + d['graph'][-1] + '.pkl'
        with open(infile, 'rb') as handle:
            g = pickle.load(handle)

        assert(n == len(g))

        median_g = str_list_to_median_graph(len(g), sampler.res['SAMPLES'][burnin:], .5)

        TP, TN, FP, FN = get_accuracies(g, median_g)

        d['TP'].append(TP)
        d['TN'].append(TN)
        d['FP'].append(FP)
        d['FN'].append(FN)

    df = pd.DataFrame(d)
    df.to_csv('results/combined_summary.csv', index=False)

if __name__ == "__main__":
    main()
