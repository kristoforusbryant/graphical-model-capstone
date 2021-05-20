import argparse

class Parser:
    def __init__(self, sys_args):
        """[Parser function to allow command line interface]
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, help="path to file containing the observation matrix", required=True)
        parser.add_argument('--outfile', type=str, help="path to file that will contain the result", required=True)
        parser.add_argument('--summarise', action='store_true', help="path to file that contains summaries about the result")
        parser.add_argument('--basis', type=str, help="'edge' or 'cycle' basis", default ='cycle')
        parser.add_argument('--treeprior', type=str, help="spanning tree that generates the cycle basis ('all', 'star', or 'path')",
                             default ='all')
        parser.add_argument('-r', type=int, help="success parameter of the negative binomial hyperprior", default=6)
        parser.add_argument('-p', type=float, help="probability parameter of the negative binomial hyperprior", default=.75)
        parser.add_argument('--cob-freq',  type=int, help="frequency of change of basis", default=100)
        parser.add_argument('--seed', type=int, help="random number seed set at the start of the MCMC", default=None)
        self.args = parser.parse_args(sys_args)
