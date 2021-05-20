import numpy as np
def is_similar(g1, g2, threshold=5):
    return np.sum(np.abs(g1.GetBinary - g2.GetBinary)) < threshold