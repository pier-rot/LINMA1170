import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import numba

start = time.time()

# Approximation des nombres de conditionement
# du problème des moindres carrés en fonction de A et de b.
@numba.jit(nopython=True, parallel=True)
def makeDeltas(A,b,x, p=1000):

    deltas = np.zeros((p,2))
    m,n = A.shape
    for i in numba.prange(p):

        epsilon = 1e-10
        Ap = A + epsilon * np.random.randn(m,n)
        bp = b + epsilon * np.random.randn(m)

        xAp = np.linalg.lstsq(Ap, b)[0]
        xbp = np.linalg.lstsq(A, bp)[0]

        deltas[i,0] = norm(((xAp - x)/norm(x))/(norm(Ap - A, ord=2)/ norm(A)))
        deltas[i,1] = norm(((xbp - x)/norm(x))/(norm(bp - b, ord=2)/ norm(b)))
    return (np.max(deltas[:,0]),np.max(deltas[:,1]))

t1 = time.time()
makeDeltas(np.random.randn(3,2), np.random.randn(3), np.random.randn(2))
t2 = time.time()
makeDeltaCompileTime = t2-t1

@numba.jit(nopython=True, parallel=True)
def makeCondErrors (d=10):
    ACondErrors = np.zeros((d,))
    bCondErrors = np.zeros((d,))
    ns = np.zeros((d,))
    ms = np.zeros((d,))

    for i in numba.prange(d):
        n = (i+1)*10
        m = 3*n
        n = i+1
        m = 3*n
        ns[i] = n
        ms[i] = m
        A = np.random.randn(m,n)
        b = np.random.randn(m)
        x = np.linalg.lstsq(A,b)
        x = x[0]
        normA = norm(A, ord=2)
        normb = norm(b)
        normx = norm(x)
        CondA = np.linalg.cond(A)
        y = A@x
        eta = (normA * normx)/norm(y)
        theta = np.arccos(norm(y)/normb)
        Cond_x_b = CondA / (eta*np.cos(theta))
        Cond_x_A = CondA + ((CondA*CondA)*np.tan(theta))/eta
        ACondErrors[i], bCondErrors[i] = makeDeltas(A,b,x)
        ACondErrors[i] = abs(ACondErrors[i] - Cond_x_A)
        bCondErrors[i] = abs(bCondErrors[i] - Cond_x_b)
    return ACondErrors,bCondErrors,ms,ns
t1 = time.time()
makeCondErrors(1)
t2 = time.time()
makeCondCompileTime = t2 - t1
t1 = time.time()
AErrors,bErrors,ms,ns = makeCondErrors(100)
t2 = time.time()
makeCondComputeTime = t2- t1


fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle("Nombres de conditionement pour le problème des moindres carrés")
ax1.set_title("Erreurs du nombre de conditionement pour des perturbations de A")
ax1.set(xlabel = "Taille de A", ylabel="Erreurs d'approximation du nombre de contionement")
ax1.scatter(np.multiply(ms,ns),AErrors)
ax2.set_title("Erreurs du nombre de conditionement pour des perturbations de b")
ax2.label_outer()
ax2.set(xlabel = "Taille de b")
ax2.scatter(ms, bErrors)
stop = time.time()
print(f"makeDeltasCompileTime = {makeDeltaCompileTime} ; \nmakeCondCompileTime = {makeCondCompileTime} ; \nmakeCondComputeTime = {makeCondComputeTime} ; \ntotalExecutionTime = {stop - start}")
plt.show()

