import numpy as np
import numba
import matplotlib.pyplot as plt
import time

@numba.jit(nopython=True, parallel=False)
def lu_factorization(A):
    m, n = A.shape

    L = np.eye(m)
    U = A.copy()

    for k in numba.prange(m - 1):

        for i in numba.prange(k + 1, m):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= (factor * U[k, k:])

    return L, U
@numba.jit(nopython=True, parallel=True)
def para_lu_factorization(A):
    m, n = A.shape

    L = np.eye(m)
    U = A.copy()

    for k in numba.prange(m - 1):

        for i in numba.prange(k + 1, m):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= (factor * U[k, k:])

    return L, U

# Dummy example to compile code with numba :
A = np.array([[4, 3], [6, 3]], dtype=float)
x = np.array([1,2], dtype=float)
t1 = time.time()
L, U = lu_factorization(A)
L, U = para_lu_factorization(A)
t2 = time.time()
CompileTime = t2 - t1


ns = np.arange(100, 2000, 50)
times = np.zeros((ns.size))
paraTimes = np.zeros((ns.size))
for i in range(ns.size):
    A = np.random.randn(ns[i],ns[i])
    t1 = time.time()
    L, U = lu_factorization(A)
    t2 = time.time()
    times[i] = t2-t1
    t1 = time.time()
    L, U = para_lu_factorization(A)
    t2 = time.time()
    paraTimes[i] = t2-t1


plt.loglog(ns, times, label="Complexité single thread", color="blue")
plt.loglog(ns, paraTimes, label="Complexité mutli-thread", color="yellow")
plt.loglog(ns, 2/3 * ns**3 * 1e-9,color="orange", label="2/3 * n^3")
plt.title("Complexité temporelle de la décomposition LU")
plt.xlabel("Echelle log de l'ordre n de A")
plt.ylabel("Echelle log du temps de calcul de la décomposition LU")
plt.legend()
plt.show()

print(f"Compile = {CompileTime}, Compute = {np.sum(times) + np.sum(paraTimes)}")
