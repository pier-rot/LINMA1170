import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp
import numba

@numba.jit(nopython=True, parallel=False)
def cholesky_crout(A):
    C = A
    m,n = A.shape

    for k in range(n):
        for j in range(k+1, n):
            for i in range(j,n):
                C[i,j] = C[i,j] - C[i,k]*C[j,k]/C[k,k]
        
        for i in range(k,n):
            C[i,k] = C[i,k]/np.sqrt(C[k,k])
    return C

n=10
m= 15
Cd = cholesky_crout(np.random.randn(2,2))
x = np.random.randn(n)
A = np.random.randn(m,n)
A = A.T @ A
L = cholesky_crout(A)
lnp = np.linalg.cholesky(A)
print(A @ x)
print(lnp @ lnp.T @ x)