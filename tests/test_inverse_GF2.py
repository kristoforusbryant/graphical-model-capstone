import numpy as np
from utils.inverse_binary import inverse_GF2, GF2

def test_inverse_GF2():
    n = 50
    count = 0
    np.random.seed(123)
    while count < 50:
        A = np.random.randint(0,2, (n, n))
        if np.linalg.det(GF2(A)):
            A_inv = inverse_GF2(A)
            assert((GF2(A) @ A_inv == np.eye(n, dtype=int)).all())
            count += 1