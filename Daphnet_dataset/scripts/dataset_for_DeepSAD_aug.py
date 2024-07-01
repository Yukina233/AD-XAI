import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

# 针对故障检测数据集，构建训练集和测试集用于DeepSAD
path_project = '/home/yukina/Missile_Fault_Detection/project'

dataset_name = 'window=100, step=10'
augment_param = 'no_GAN, std, window=100, step=10, no_tau2_K=7,deepsad_epoch=50,gan_epoch=20,lam1=0.99,lam2=0.01,tau1=1'
iterations = [0, 1, 2, 3]
# Create the dataset
root_dir = os.path.join(path_project, 'data/Daphnet/yukina_data')
# Save as NPY

# read original train data
DeepSAD_data = np.load(
    os.path.join(root_dir, f'DeepSAD_data, {dataset_name}/train.npz'))
X_train = DeepSAD_data['X_train']
y_train = DeepSAD_data['y_train']

for iteration in iterations:
    # read augment data
    ensemeble_data_dir = os.path.join(root_dir, f'ensemble_data, {dataset_name}/augment', augment_param, f'{iteration}')

    X_aug = []
    y_aug = []
    for cluster_data_id in os.listdir(ensemeble_data_dir):
        cluster_data = np.load(os.path.join(ensemeble_data_dir, cluster_data_id))
        X_cluster_aug = cluster_data['X_train'][np.where(cluster_data['y_train'] == 1)]
        y_cluster_aug = np.ones(X_cluster_aug.shape[0])
        X_aug.append(X_cluster_aug)
        y_aug.append(y_cluster_aug)

    X_aug = np.concatenate(X_aug)
    y_aug = np.concatenate(y_aug)

    output_path = os.path.join(path_project, root_dir, 'DeepSAD_aug_data', augment_param, f'{iteration}')
    os.makedirs(output_path, exist_ok=True)

    np.savez(os.path.join(output_path, f"train.npz"), X=[0], y=[0], X_train=np.concatenate((X_train, X_aug)),
             y_train=np.concatenate((y_train, y_aug)),
             X_test=[0], y_test=[0])

    print('Data generate complete.')
