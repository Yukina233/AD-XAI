path_project = '/home/yukina/Missile_Fault_Detection/project/'
from spot import dSPOT
import scipy.io as scio
import numpy as np
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding,Isomap,TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA,PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import pandas as pd
import matplotlib.ticker as mtick
import ctypes as c
import time
from keras.models import load_model

def get_dat(path):
    a = pd.read_csv(path,delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b

path1 = path_project + 'anomaly_detection/data/zc1.dat'
normal_data = get_dat(path1)
path2 = path_project + 'anomaly_detection/data/ks2_7.dat'
ka_data = get_dat(path2)
path3 = path_project + 'raw/ks/2/ks2_2.dat'
new_data = get_dat(path3)

# Visualize normal data
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
count = 0
for i in range(3):
    for j in range(5):
        if count < 13:
            axes[i][j].plot(normal_data[:5000, i*5+j])
            count += 1
        else:
            break
plt.title('Normal Data')
plt.tight_layout()
plt.show()


# Visualize ks data
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
count = 0
for i in range(3):
    for j in range(5):
        if count < 13:
            axes[i][j].plot(ka_data[:5000, i*5+j])
            count += 1
        else:
            break
plt.title('Kasi Data')
plt.tight_layout()
plt.show()

# Visualize ks data
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
count = 0
for i in range(3):
    for j in range(5):
        if count < 13:
            axes[i][j].plot(new_data[:5000, i*5+j])
            count += 1
        else:
            break
plt.title('New Data')
plt.tight_layout()
plt.show()
print('Finished!')