import torch
import torch.nn as nn
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

import pythae_modified
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from pythae_modified.models.nn import BaseEncoder, BaseDecoder
from pythae_modified.models.base.base_utils import ModelOutput
from pythae_modified.models.vq_vae.vq_vae_config import Simplified_VQVAEConfig
from pythae_modified.models.vq_vae.vq_vae_model import Simplified_VQVAE
from pythae_modified.models import VQVAE, VQVAEConfig

path_project = '/home/yukina/Missile_Fault_Detection/project/'


def get_dat(path):
    a = pd.read_csv(path, delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b


path = path_project + 'anomaly_detection/data/ks2_7.dat'
data = get_dat(path)
data_to_plot = data.transpose()

# data visualization
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
fig.suptitle('Data Visualization')
for i in range(3):
    if i != 2:
        for j in range(5):
            axes[i][j].plot(data_to_plot[i * 5 + j])
            axes[i][j].set_title('index: ' + str(i * 5 + j))
    else:
        for j in range(3):
            axes[i][j].plot(data_to_plot[i * 5 + j])
            axes[i][j].set_title('index: ' + str(i * 5 + j))
plt.tight_layout()
plt.show()
# 打印每个变量的值
