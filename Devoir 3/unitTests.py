import devoir3 as d3
from numpy import dot
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
c_type = d3.c_type

n = 5
m = 10
p = 4
x = np.random.randn(m)
y = np.random.randn(n)
z = np.random.randn(m)
A = np.random.randn(m,m).astype(c_type)
B = np.random.randn(n,p)
S = np.random.randn(m,m)

#print("++++++++++++ VM_prod ++++++++++++")
#print(d3.dot(x,A))
#print(x @ A)
#
#print("++++++++++++ MV_prod ++++++++++++")
#print(d3.dot(A,y))
#print(A @ y)
#
#print("++++++++++++ vdot ++++++++++++")
#print(d3.dot(x,z))
#print(x @ z)
#
#print("++++++++++++ MM_prod ++++++++++++")
#print(d3.dot(A,B))
#print(A @ B)

#print(d3.hessenberg(S, np.zeros((m,m))))
#I = np.eye(m,m, dtype=c_type)
#P = np.eye(m,m, dtype=c_type)
#A_or = A.copy()
for k in range(m-2):
    x = A[k+1:,k]
    e1 = np.zeros_like(x,dtype=c_type)
    e1[0] = 1
    vk = np.sign(x[0])*d3.vnorm(x)*e1 + x
    vk = vk/d3.vnorm(vk)
    Pk = I[k+1:, k+1:] - 2*d3.outer(vk, np.conjugate(vk))
    P[k+1:, k+1:] = d3.dot(P[k+1:, k+1:], Pk)
    A[k+1:,k:] = d3.dot(Pk, A[k+1:, k:])
    A[:,k+1:] = d3.dot(A[:,k+1:], Pk)

Acopy = A.copy()
d3.hessenberg(A, np.eye(m,m).astype(c_type))
fig,axs = plt.subplots(1,2)
axs[0].imshow(np.abs(A), cmap='Blues')
axs[1].imshow(np.abs(Acopy), cmap="Blues")

plt.show()