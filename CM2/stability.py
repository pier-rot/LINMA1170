import numpy as np

m = 100; n = 50

# On génère Q et R, et on calcule A = QR
R = np.triu(np.random.randn(n,n))
Q,_ = np.linalg.qr(np.random.randn(m,n))
A = Q @ R

# Q,R: f(~y)

# On calculer la décomposition QR de A
Q2, R2 = np.linalg.qr(A) # ~f(y)

# Q2, R2: ~f(y)

# Est-ce que l'algo est stable ?
print(f'{np.linalg.norm(Q2-Q) / np.linalg.norm(Q) = }')
print(f'{np.linalg.norm(R2-R) / np.linalg.norm(R) = }')

# Est-ce que l'algo est inversement stable ?
print(f'{np.linalg.norm(A - Q2 @ R2) / np.linalg.norm(A) = }')
# ||~y - y|| / ||y|| <= o(eps)
