#%%
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

#%%

H, x = loadmat("data/H.mat")["H"], loadmat("data/x.mat")["x"]
# %%
y = H @ x + np.random.normal(0, 1, size=(H.shape[0], 1))
# %%
plt.imshow(y.reshape((90, 180), order="F"))
plt.savefig("y.png")
# %%
plt.imshow(x.reshape((90, 90), order="F"))
plt.savefig("x.png")
# %%
G = loadmat("data/G.mat")["G"]
# %%
