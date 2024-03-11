import numpy as np
import numba
import matplotlib.pyplot as plt
import time

@numba.jit(nopython=True, parallel=False, cache=True)
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
@numba.jit(nopython=True, parallel=True, cache=True)
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


@numba.jit(nopython=True, parallel=True, cache=True)
def cholesky(A):
    m,n = A.shape
    if (m != n):
        raise ValueError("A must be square")
    C = A
    for k in numba.prange(m):
        for j in numba.prange(k+1, m):
            C[j,j:] -= C[k,j:]*C[k,j]/C[k,k]
        C[k,k:] /= np.sqrt(C[k,k])
    return C

# Dummy example to compile code with numba :
A = np.array([[4, 3], [6, 3]], dtype=float)
x = np.array([1,2], dtype=float)
t1 = time.perf_counter()
L, U = lu_factorization(A)
L, U = para_lu_factorization(A)
t2 = time.perf_counter()
CompileTime = t2 - t1
A = A.T @ A
C = cholesky(A)

# Computing data for graphs
ns = np.arange(100, 2000, 50)
times = np.zeros((ns.size))
paraTimes = np.zeros((ns.size))
choTimes = np.zeros((ns.size))
for i in range(ns.size):
    A = np.random.randn(ns[i],ns[i])
    t1 = time.perf_counter_ns()
    L, U = lu_factorization(A)
    t2 = time.perf_counter_ns()
    times[i] = t2-t1
for i in range(ns.size):
    A = np.random.randn(ns[i],ns[i])
    t1 = time.perf_counter_ns()
    L, U = para_lu_factorization(A)
    t2 = time.perf_counter_ns()
    paraTimes[i] = t2-t1
    
    A = A.T @ A
    t1 = time.perf_counter_ns()
    C = cholesky(A)
    t2 = time.perf_counter_ns()
    choTimes[i] = t2-t1

gain = np.multiply(paraTimes, 1/choTimes)


plt.loglog(ns, times, label="Complexité single thread", color="blue")
plt.loglog(ns, paraTimes, label="Complexité mutli-thread", color="yellow")
plt.loglog(ns, 1/3 * ns**3,color="orange", label="1/3 * n^3")
plt.loglog(ns, choTimes, label="Complexité Cholesky", color="green")
plt.title("Complexité temporelle de la décomposition LU")
plt.xlabel("Echelle log de l'ordre n de A")
plt.ylabel("Echelle log du temps de calcul de la décomposition LU en nanosecondes")
plt.legend()
plt.show()

plt.plot(ns, gain, label="Gain de temps de Cholesky par rapport à décomposition LU")
plt.title("Complexité temporelle de l'algorithme de Cholesky")
plt.xlabel("Taille n de matrice A n x n")
plt.ylabel("Gain par rapport à la décomposition LU")
plt.show()

#print(f"Compile = {CompileTime}, Compute = {(np.sum(times) + np.sum(paraTimes)+ np.sum(choTimes))*1e-9}")
