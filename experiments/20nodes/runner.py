from collections import OrderedDict
import numpy as np
from utils.config_utils import run_config, get_config_l, config_to_path
from multiprocessing import Pool
import pickle

config = OrderedDict({  'n': [20],
                        'n_obs': [2],
                        'init': ['empty', 'fixed'], #'complete',
                        'true_graph': ["empty", "circle", "random0", "random1", "random2", "random3"],
                        'prior': ['basis-count'],
                        'basis': ['hub', 'uniform', 'edge'],
                        'proposal': ['naive'],
                        'cob_freq': [100],
                        'iter': [int(1e4)],
                        'seed': 123 })


def run(conf):
    n, n_obs = conf['n'], conf['n_obs']
    name = conf['true_graph']
    data = np.loadtxt(f"data/{name}_{n}_{n_obs}.dat", delimiter=',')
    sampler = run_config(data, conf)

    with open(f"data/graph_{conf['true_graph']}_{conf['n']}_{conf['n_obs']}.pkl", 'rb') as handle:
        g = pickle.load (handle)

    for burnin in [0, int(.1 * sampler.iter), int(.25 * sampler.iter)]:
        print(f"saving to {config_to_path(conf)[:-4]}_burnin-{burnin}.short")
        with open(config_to_path(conf)[:-4] + f"_burnin-{burnin}.short", 'wb') as handle:
            pickle.dump(sampler.get_summary(g), handle)

pool = Pool()
pool.map(run, get_config_l(config))
