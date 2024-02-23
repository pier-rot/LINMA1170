import numpy as np
import numba
import time
import matplotlib.pyplot as plt

# ======== Function Definitions ========

def norm(array = np.array([[]])):
    # Norme Frobenius d'une matrice
    squaredArr = np.sum(np.power(array,2).flatten())

    return np.sqrt(squaredArr)

def make_t(m,P):
    # Construction du tableau de paramètre t_i
    if (m != P.shape[0]):
        raise ValueError(f"P and m are different sizes; \nm = {m}, P.shape[0] = {P.shape[0]}")
    t = np.zeros(m)
    ropeLength = 0
    for i in range(1,m):
        ropeLength += norm(P[i] - P[i-1])
    t[m-1] = 0
    for i in range(1,m):
        t[i] = t[i-1] + (norm(P[i] - P[i-1]))/ropeLength
    return t

def make_T(m,n,t):
    # Construction du tableau de points de contrôle
    T = np.zeros(n+4)
    d = m/(n-3)
    for j in range(1, n-3):
        i = int((j*d) // 1)
        alpha = (j*d) - i
        T[j+3] = (1 - alpha)*t[i-1] + alpha*t[i]
    T[n:] = 1

    return T


@numba.jit(nopython=True, parallel=False)
def qr(A):
    # Décomposition QR de la matrice A

    
    return A

def lstsq(A,B):
    
    return X


# Data
m = 400
n = 30
