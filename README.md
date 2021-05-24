# cbMCMC

An efficient MCMC algorithm for graphical model inference using cycle-basis prior.

## Basis Usage
```python
>>> from utils.cbmcmc import cbmcmc

>>> import numpy as np

>>> mu, Sigma = (np.zeros(10), np.eye(10))

>>> data = np.random.multivariate_normal(mu, Sigma, 1000) 

>>> sampler = cbmcmc(data, it=200, basis='cycle', treeprior='all')

>>> res = sampler.res['SAMPLES']
```
