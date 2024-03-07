import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

n = 2
m = 3
A = np.random.rand(m,n)
b = np.random.rand(m)
x = np.linalg.lstsq(A,b, rcond=None)[0]
normA = norm(A, ord=2)
normb = norm(b)
normx = norm(x)

# Conditioning
CondA = np.linalg.cond(A)
y = A@x
eta = (normA * normx)/norm(y)
theta = np.arccos(norm(y)/normb)
# Conditioning numbers
Cond_x_b = CondA / (eta*np.cos(theta))
Cond_x_A = CondA + ((CondA*CondA)*np.tan(theta))/eta

# Pertubations of A and b
p = 1000
deltas = np.zeros((p,2))
for i in range(p):

    epsilon = 1e-10
    Ap = A + epsilon * np.random.randn(m,n)
    bp = b + epsilon * np.random.randn(m)

    xAp = np.linalg.lstsq(Ap, b)[0]
    xbp = np.linalg.lstsq(A, bp)[0]
    
    deltas[i,0] = norm(((xAp - x)/normx)/(norm(Ap - A, ord=2)/ normA))
    deltas[i,1] = norm(((xbp - x)/normx)/(norm(bp - b, ord=2)/ normb))


myCond_x_A = np.max(deltas[:,0])
myCond_x_b = np.max(deltas[:,1])
print(f"Cond_x_b = {Cond_x_b}, myCond_x_b = {myCond_x_b}\nCond_x_A = {Cond_x_A}, myCond_x_A = {myCond_x_A}")