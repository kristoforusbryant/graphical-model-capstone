import numpy as np
from scipy import special

def chooseln(N, k):
    return special.gammaln(N+1) - special.gammaln(N-k+1) - special.gammaln(k+1)

class TruncatedNB:
    # Assume Finite Support
    def __init__(self, r, p):
        self._r = r
        self._p = p

    def __call__(self, size):
        return chooseln(size + self._r - 1, size) + self._r * np.log(1-self._p) + size * np.log(self._p)

class Uniform:
    # Assume Finite Support
    def __init__(self, params=None):
        pass

    def __call__(self, size):
        return 1

