from multiprocessing import Pool
from utils.cbmcmc import cbmcmc

edgeIO = [ ('data/empty.dat', 'data/graph_empty.pkl', 'results/edge_empty_samples.pkl'),
           ('data/circle.dat', 'data/graph_circle.pkl', 'results/edge_circle_samples.pkl'),
           ('data/random0.dat', 'data/graph_random0.pkl', 'results/edge_random0_samples.pkl'),
           ('data/random1.dat', 'data/graph_random1.pkl', 'results/edge_random1_samples.pkl'),
           ('data/random2.dat', 'data/graph_random2.pkl', 'results/edge_random2_samples.pkl'),
           ('data/random3.dat', 'data/graph_random3.pkl', 'results/edge_random3_samples.pkl'),
           ('data/complete.dat', 'data/graph_complete.pkl', 'results/edge_complete_samples.pkl'), ]

allIO = [ ('data/empty.dat', 'data/graph_empty.pkl', 'results/all_empty_samples.pkl'),
           ('data/circle.dat', 'data/graph_circle.pkl', 'results/all_circle_samples.pkl'),
           ('data/random0.dat', 'data/graph_random0.pkl', 'results/all_random0_samples.pkl'),
           ('data/random1.dat', 'data/graph_random1.pkl', 'results/all_random1_samples.pkl'),
           ('data/random2.dat', 'data/graph_random2.pkl', 'results/all_random2_samples.pkl'),
           ('data/random3.dat', 'data/graph_random3.pkl', 'results/all_random3_samples.pkl'),
           ('data/complete.dat', 'data/graph_complete.pkl', 'results/all_complete_samples.pkl'),]

hubIO = [ ('data/empty.dat', 'data/graph_empty.pkl', 'results/hub_empty_samples.pkl'),
           ('data/circle.dat', 'data/graph_circle.pkl', 'results/hub_circle_samples.pkl'),
           ('data/random0.dat', 'data/graph_random0.pkl', 'results/hub_random0_samples.pkl'),
           ('data/random1.dat', 'data/graph_random1.pkl', 'results/hub_random1_samples.pkl'),
           ('data/random2.dat', 'data/graph_random2.pkl', 'results/hub_random2_samples.pkl'),
           ('data/random3.dat', 'data/graph_random3.pkl', 'results/hub_random3_samples.pkl'),
           ('data/complete.dat', 'data/graph_complete.pkl', 'results/hub_complete_samples.pkl'),]

pathIO = [ ('data/empty.dat', 'data/graph_empty.pkl', 'results/path_empty_samples.pkl'),
           ('data/circle.dat', 'data/graph_circle.pkl', 'results/path_circle_samples.pkl'),
           ('data/random0.dat', 'data/graph_random0.pkl', 'results/path_random0_samples.pkl'),
           ('data/random1.dat', 'data/graph_random1.pkl', 'results/path_random1_samples.pkl'),
           ('data/random2.dat', 'data/graph_random2.pkl', 'results/path_random2_samples.pkl'),
           ('data/random3.dat', 'data/graph_random3.pkl', 'results/path_random3_samples.pkl'),
           ('data/complete.dat', 'data/graph_complete.pkl', 'results/path_complete_samples.pkl'),]

it = 10000
r, p = (1, .5)
cob_freq = 100
seed = 123

# Edge
def edge_runner(t):
    d, i, o = t
    cbmcmc(d, it, 'edge', r=r, p=p, outfile=o, init=i)
    return 0
pool = Pool()
pool.map(edge_runner, edgeIO)

# All
# def all_runner(t):
#     d, i, o = t
#     cbmcmc(d, it, 'cycle', 'all', r, p, cob_freq, outfile=o, init=i)
#     return 0
# pool = Pool()
# pool.map(all_runner, allIO)

# # Hub
# def hub_runner(t):
#     d, i, o = t
#     cbmcmc(d, it, 'cycle', 'hub', r, p, cob_freq, outfile=o, init=i)
#     return 0
# pool = Pool()
# pool.map(hub_runner, hubIO)

# # Path
# def path_runner(t):
#     d, i, o = t
#     cbmcmc(d, it, 'cycle', 'path', r, p, cob_freq, outfile=o, init=i)
#     return 0
# pool = Pool()
# pool.map(path_runner, pathIO)

