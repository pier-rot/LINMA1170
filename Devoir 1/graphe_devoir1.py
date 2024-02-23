import numpy as np
import time
import matplotlib.pyplot as plt
from devoir1 import *
# Étude de la complexité de la décomposition QR

# Caching de qr et lstsq
dummyA = np.random.rand(2,2)
dummyB = np.random.rand(2,2)
lstsq(dummyA,dummyB)


def computeTime(m, n):
	# Fonction de calcul des temps en fonction de m et n
	deltaT = np.zeros(m.shape)
	for i in range(m.shape[0]):
		for j in range(m.shape[1]):
			A = np.random.rand(i+1,j+1)
			t1 = time.time()
			C = qr(A)
			t2 = time.time()
			deltaT[i,j] = t2 - t1

	return deltaT

# Definition des m et n à évaluer
m_values = np.arange(10, 2000, 40)
n_values = np.arange(10, 2000, 40)

# Meshgrid de m_values et n_values
m, n = np.meshgrid(m_values, n_values)

# Calcul du mesh des temps de calcul
z = computeTime(m,n)

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot la surface
ax.plot_surface(m, n, z, cmap='viridis')

# Label et titre
ax.set_xlabel('m')
ax.set_ylabel('n')
ax.set_zlabel('Temps de la factorisation QR')

# Show the plot
plt.show()