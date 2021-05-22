from scipy.stats import nbinom

class TruncatedNB:
    # Assume Finite Support
    def __init__(self, r, p):
        self._r = r
        self._p = p
        self._rv = nbinom(self._r, self._p)

    def __call__(self, size):
        return self._rv.pmf(size)

class Uniform:
    # Assume Finite Support
    def __init__(self, params=None):
        pass

    def __call__(self, size):
        return 1

