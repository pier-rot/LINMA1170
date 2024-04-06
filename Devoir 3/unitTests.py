from devoir3 import dot, mprod
import numpy as np

a = np.array([1,2,3])
b = np.array([1,2,3])

print(dot(a,b))

A = np.array([[1,0]])
A2 = np.array([1,0])
B = np.array([[1,0],
              [0,1]])
print(mprod(A,B))
print(A.shape)
print(A2.reshape((1,A2.size)).shape)