import numpy as np
import numba
import time
import matplotlib.pyplot as plt

# ======== Function Definitions ========

def myNorm(array = np.array([[]])):
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
		ropeLength += myNorm(P[i] - P[i-1])
	t[m-1] = 0
	for i in range(1,m):
		t[i] = t[i-1] + (myNorm(P[i] - P[i-1]))/ropeLength
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

def backward_substitution(R, b):
    # Substitution arrière
    n = len(b)
    x = np.zeros_like(b, dtype=float)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]

    return x

def b(t,T,i,p):
	# Coefficient de B-spline d'ordre p, au paramètre t et de noeuds T
	if p == 0:
		return (T[i] <= t)*(t < T[i+1])
	else:
		u  = 0.0 if T[i+p ]  == T[i]   else (t-T[i])/(T[i+p]- T[i]) * b(t,T,i,p-1)
		u += 0.0 if T[i+p+1] == T[i+1] else (T[i+p+1]-t)/(T[i+p+1]-T[i+1]) * b(t,T,i+1,p-1)
		return u
@numba.jit(nopython=True, parallel=False)
def qr(A):
	# Décomposition QR de la matrice A
	m,n = A.shape
	Q = A.copy()
	R = np.zeros((n,n))
	for k in range(n):
		kNorm = 0
		for l in range(Q[:,k].size):
			kNorm+= Q[:,k][l]**2
		
		R[k,k] = np.sqrt(kNorm)
		Q[:,k] /= R[k,k]
		for j in range(k+1,n):
			R[k,j] = Q[:,k] @ Q[:,j]
			Q[:,j] -= R[k,j] * Q[:,k]
	return Q,R

def lstsq(A,B):
	# Résolution du système des moindres carrés donné par : A^T A X = A^T B
	# en utilisant une décomposition QR
	Q,R = qr(A)
	V = Q.T @ B
	m,p = V.shape
	_,n = R.shape
	X = np.zeros((n,p))
	for i in range(p):
		X[:,i] = backward_substitution(R,B[:,i])
	return X


# Data
m = 400
n = 30

A = np.random.rand(m,n)
Q,R = qr(A)
fig,axs = plt.subplots(1,2)
axs[0].imshow(Q.T @ Q, cmap='Blues')
axs[1].imshow(np.abs(R), cmap='Blues')

plt.show()