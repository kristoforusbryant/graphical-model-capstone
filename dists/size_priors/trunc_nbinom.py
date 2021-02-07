from scipy.stats import nbinom 

# Truncated Negative Binomial for Sizes
class SizePrior: 
    # Assume Finite Support
    def __init__(self, params):
        self._r = params['r'] 
        self._p = params['p']
        self._rv = nbinom(self._r, self._p)
        
    def __call__(self, size):
        return self._rv.pmf(size)