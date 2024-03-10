import numpy as np
import matplotlib.pyplot as plt

A = np.random.randn(5,3)

# L = np.tril(A)
# U = np.triu(A)
# 
# ltl = L.T @ L
# utu = U.T @ U
# ll = L @ L
# uu = U @ U
# 
# fig, axs = plt.subplots(2,2)
# axs[0][0].imshow(ltl, cmap="Blues", label="ltl")
# axs[0][0].set_title("L.T @ L")
# axs[0][1].imshow(utu, cmap="Blues", label="utu")
# axs[0][1].set_title("U.T @ U")
# axs[1][0].imshow(ll, cmap="Blues", label="ll")
# axs[1][0].set_title("L @ L")
# axs[1][1].imshow(uu, cmap="Blues", label="uu")
# axs[1][1].set_title("U @ U")


plt.imshow(A.T @ A, cmap="Blues")
plt.show()