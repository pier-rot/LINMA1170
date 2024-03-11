import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp
import numba

A = np.random.randn(300,10)
print((A.T @ A).shape)