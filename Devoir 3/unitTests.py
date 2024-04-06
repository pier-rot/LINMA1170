import devoir3 as d3
import numpy as np

n = 5
m = 3
p = 4
x = np.random.randn(m)
y = np.random.randn(n)
z = np.random.randn(m)
A = np.random.randn(m,n)
B = np.random.randn(n,p)

#d3.mprod(A,x)
print("++++++++++++ VM_prod ++++++++++++")
print(d3.dot(x,A))
print(x @ A)

print("++++++++++++ MV_prod ++++++++++++")
print(d3.dot(A,y))
print(A @ y)

print("++++++++++++ vdot ++++++++++++")
print(d3.dot(x,z))
print(x @ z)

print("++++++++++++ MM_prod ++++++++++++")
print(d3.dot(A,B))
print(A @ B)

#print(d3.MM_prod(A,B))
#print(A @ B)