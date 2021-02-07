import numpy as np
# utils should be inside the path of the file that imports this

class Prior:
    def __init__(self, n, Param, basis, prob_c, prob_s):
        # prob_c: distribution of the number of bases i.e. "count"
        # prob_s: distribution of the basis sizes
        self._n = n
        self._Param = Param 
        self._basis = basis 
        self._prob_c = prob_c # for 0, ... number of basis
        self._prob_s = prob_s # for 0, ... m 
        
        
    def Sample(self):
        param = self._Param(self._n, basis=self._basis)
        
        # the count is distributed as p_c 
        unscaled_p_c = np.array([self._prob_c(k) for k in range(len(self._basis) + 1)])
        p_c = unscaled_p_c / np.sum(unscaled_p_c)
        count = np.random.choice(range(len(self._basis) + 1), p=p_c)
        
        # the sizes of the bases are distributed as p_s 
        unscaled_p_s = np.array([self._prob_s(b.EdgeCount()) for b in self._basis])
        p_s = unscaled_p_s / np.sum(unscaled_p_s)
        idx = np.random.choice(np.arange(len(p_s)), p=p_s, size=count, replace=False) 
        for i in idx: 
            param.BinAddOneBasis(i)
            
        return param
    
    
    # This PDF is unnormalised over the span of the basis 
    def PDF(self, param):
        log_p_c = np.log(self._prob_c(param.EdgeCount()))
        log_p_s = 0 
        for i in range(len(param._basis)): 
            if param._basis_active[i]:
                k = param._basis[i].EdgeCount()
                log_p_s += np.log(self._prob_s(k))
        return log_p_c + log_p_s 
    
    def ParamType(self): 
        return self._Param.__name__