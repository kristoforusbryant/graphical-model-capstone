import numpy as np
from utils.laplace_approximation import constrained_cov
from utils.G_Wishart import G_Wishart
# works as long as Param has .GetDOL() function and ._dol property

class GW:
    def __init__(self, data, delta, D, Param):
        param = Param(D.shape[0])
        # GW: G-Wishart(G, delta, D)
        self.GW_prior = G_Wishart(param, delta, D)
        self._D = D
        self._U = data.transpose() @ data
        D_star = D + self._U
        self.GW_posterior = G_Wishart(param, delta + data.shape[0], D_star)
        self._Param = Param

    def Move(self, param_):
        self.GW_prior.G.SetFromG(param_)
        self.GW_posterior.G.SetFromG(param_)
        D_star = self._D + self._U
        self.GW_posterior.SetD(D_star)

    def PDF(self, param_=None):
        if param_ is not None:
            self.Move(param_)
        return self.GW_posterior.IG() - self.GW_prior.IG()# as log prob

    def ParamType(self):
        return self._Param.__name__

class GW_LA:
    def __init__(self, data, delta, D, Param):
        param = Param(D.shape[0])
        # GW: G-Wishart(G, delta, D)
        self.GW_prior = G_Wishart(param, delta, D)
        self._D = D
        self._U = data.transpose() @ data
        D_star = D + self._U
        self.GW_posterior = G_Wishart(param, delta + data.shape[0], D_star)
        self._Param = Param

    def Move(self, param_):
        self.GW_prior.G.SetFromG(param_)
        self.GW_posterior.G.SetFromG(param_)
        D_star = self._D + self._U
        self.GW_posterior.SetD(D_star)

    def PDF(self, param_=None):
        if param_ is not None:
            self.Move(param_)
        return self.GW_posterior.IG_LA() - self.GW_prior.IG_LA()# as log prob

    def ParamType(self):
        return self._Param.__name__

class Delta:
    def __init__(self, param):
        self.param = param

    def PDF(self, param_):
        if self.param == param_:
            return 0
        else:
            return -np.inf

    def ParamType(self):
        return self.param.__class__