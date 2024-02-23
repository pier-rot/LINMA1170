import numpy as np
import matplotlib.pyplot as plt

def gram_schmidt(A):
	m,n = A.shape
	Q = np.zeros((m,n))
	R = np.zeros((n,n))
	for k in range(n):
		Q[:,k] = A[:,k]
		for j in range(k):
			R[j,k] = Q[:,j] @ A[:,k]
			Q[:,k] -= R[j,k] * Q[:,j]
		R[k,k] = np.linalg.norm(Q[:,k])
		Q[:,k] /= R[k,k]
	return Q,R

def modified_gram_schmidt(A):
	m,n = A.shape
	Q = A.copy()
	R = np.zeros((n,n))
	for k in range(n):
		R[k,k] = np.linalg.norm(Q[:,k])
		Q[:,k] /= R[k,k]
		for j in range(k+1,n):
			R[k,j] = Q[:,k] @ Q[:,j]
			Q[:,j] -= R[k,j] * Q[:,k]
	return Q,R

# A = np.random.randn(20,15)

eps = 1e-8
A = np.array([
	[1, 1, 1],
	[eps, 0, 0],
	[0, eps, 0],
	[0, 0, eps]
])

Q,R = modified_gram_schmidt(A)

# Q.T @ Q == I
# R est triangulaire supérieure

fig,axs = plt.subplots(1,2)
axs[0].imshow(Q.T @ Q, cmap='Blues')
axs[1].imshow(np.abs(R), cmap='Blues')
plt.show()