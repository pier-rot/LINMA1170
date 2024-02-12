import time # pour mesurer la performance de nos algorithmes
import numpy as np # pour gérer des tableaux n-d
import numba # pour accélérer des bouts de code critiques
import random #
import matplotlib.pyplot as plt

@numba.jit(nopython=True, parallel=False)
def mult(A, B):
    m = len(A)
    n = len(B)
    p = len(B[0])
    C = np.zeros((m,p))

    for i in numba.prange(m):
        for j in numba.prange(p):
            c = 0
            for k in numba.prange(n):
                c += A[i][k] * B[k][j]
            C[i][j] = c

    return C

def dumbMult(A,B):
    m = len(A)
    n = len(B)
    p = len(B[0])
    C = np.zeros((m,p))

    for i in range(m):
        for j in range(p):
            c = 0
            for k in range(n):
                c += A[i][k] * B[k][j]
            C[i][j] = c

    return C


A = np.random.rand(300,300)
B = np.random.rand(300,300)
sampleSize = 6
times = np.zeros(5)
sizes = np.array([62,125,250,500,1000])
C = mult(A,B)


for i in range(6):
    size = sizes[i]
    A = np.random.rand(size,size)
    B = np.random.rand(size,size)
    t1 = time.time()
    C = dumbMult(A,B)
    t2 = time.time()
    t12 = t2 - t1
    times[i] = t12

paraTimes = [1.77145004e-04, 6.55889511e-04, 6.34598732e-03, 8.12048912e-02, 8.75701904e-01, 1.16600738e+01]
sinTimes = [2.30073929e-04, 1.77025795e-03, 1.60620213e-02, 1.48567915e-01, 2.06024003e+00, 2.06416228e+01]

print(times)
plt.loglog(sizes, paraTimes, label = 'Parallel')
plt.loglog(sizes, sinTimes, label='JIT')
plt.loglog(sizes, times, label="Base")

# Add labels and a legend
plt.xlabel('Matrix sizes (log scale)')
plt.ylabel('Times (log scale)')
plt.title('Log-Log Plot')
plt.legend()

# Show the plot
plt.show()

