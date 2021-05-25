from multiprocessing import Pool
from utils.cbmcmc import cbmcmc

edgeIO = [ ('data/empty.dat', 'results/edge_empty_samples.pkl'),
           ('data/circle.dat', 'results/edge_circle_samples.pkl'),
           ('data/random0.dat', 'results/edge_random0_samples.pkl'),
           ('data/random1.dat', 'results/edge_random1_samples.pkl'),
           ('data/random2.dat', 'results/edge_random2_samples.pkl'),
           ('data/random3.dat', 'results/edge_random3_samples.pkl'), ]

allIO = [ ('data/empty.dat', 'results/all_empty_samples.pkl'),
           ('data/circle.dat', 'results/all_circle_samples.pkl'),
           ('data/random0.dat', 'results/all_random0_samples.pkl'),
           ('data/random1.dat', 'results/all_random1_samples.pkl'),
           ('data/random2.dat', 'results/all_random2_samples.pkl'),
           ('data/random3.dat', 'results/all_random3_samples.pkl'), ]

hubIO = [ ('data/empty.dat', 'results/hub_empty_samples.pkl'),
           ('data/circle.dat', 'results/hub_circle_samples.pkl'),
           ('data/random0.dat', 'results/hub_random0_samples.pkl'),
           ('data/random1.dat', 'results/hub_random1_samples.pkl'),
           ('data/random2.dat', 'results/hub_random2_samples.pkl'),
           ('data/random3.dat', 'results/hub_random3_samples.pkl'), ]

pathIO = [ ('data/empty.dat', 'results/path_empty_samples.pkl'),
           ('data/circle.dat', 'results/path_circle_samples.pkl'),
           ('data/random0.dat', 'results/path_random0_samples.pkl'),
           ('data/random1.dat', 'results/path_random1_samples.pkl'),
           ('data/random2.dat', 'results/path_random2_samples.pkl'),
           ('data/random3.dat', 'results/path_random3_samples.pkl'), ]

it = 10000
r, p = (5, .75)
cob_freq = 100
seed = 123

# Edge
def edge_runner(t):
    d, o = t
    cbmcmc(d, it, 'edge', outfile=o)
    return 0
pool = Pool()
pool.map(edge_runner, edgeIO)

# All
def all_runner(t):
    d, o = t
    cbmcmc(d, it, 'cycle', 'all', r, p, cob_freq, outfile=o)
    return 0
pool = Pool()
pool.map(all_runner, allIO)

# Hub
def hub_runner(t):
    d, o = t
    cbmcmc(d, it, 'cycle', 'hub', r, p, cob_freq, outfile=o)
    return 0
pool = Pool()
pool.map(hub_runner, hubIO)

# Path
def path_runner(t):
    d, o = t
    cbmcmc(d, it, 'cycle', 'path', r, p, cob_freq, outfile=o)
    return 0
pool = Pool()
pool.map(path_runner, pathIO)

