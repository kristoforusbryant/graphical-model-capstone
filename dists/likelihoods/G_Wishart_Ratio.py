import numpy as np
from utils.laplace_approximation import constrained_cov
from utils.G_Wishart import G_Wishart
# works as long as Param has .GetDOL() function and ._dol property

class Likelihood:
    def __init__(self, data, delta, D, Param):
        param = Param(D.shape[0])
        # GW: G-Wishart(G, delta, D)
        self.GW_prior = G_Wishart(param, delta, D)
        self._D = D 
        self._U = data.transpose() @ data
        D_star = constrained_cov(param.GetDOL(), D + self._U, np.eye(len(param)))
        self.GW_posterior = G_Wishart(param, delta + data.shape[0], D_star) 
        self._Param = Param
        
    def Move(self, param_):
        self.GW_prior.G.SetFromG(param_)
        self.GW_posterior.G.SetFromG(param_)
        D_star = constrained_cov(param_.GetDOL(), self._D + self._U, np.eye(len(param_)))
        self.GW_posterior.SetD(D_star)
        
    def PDF(self, param_=None):
        if param_ is not None: 
            self.Move(param_) 
        return self.GW_posterior.IG() - self.GW_prior.IG()# as log prob 
    
    def ParamType(self): 
        return self._Param.__name__