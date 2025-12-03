from numba import cuda
import numpy as np
import math

@cuda.jit
def test_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2

data = np.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(data.size / threadsperblock)
test_kernel[blockspergrid, threadsperblock](data)
print(data[0])
