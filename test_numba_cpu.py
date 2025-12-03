from numba import jit
import numpy as np

@jit(nopython=True)
def test_cpu(x):
    s = 0
    for i in range(x.size):
        s += x[i]
    return s

data = np.ones(100)
print(test_cpu(data))
