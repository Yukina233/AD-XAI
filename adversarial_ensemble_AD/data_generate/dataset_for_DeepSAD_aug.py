import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


# 针对故障检测数据集，构建训练集和测试集用于DeepSAD
path_project = '/home/yukina/Missile_Fault_Detection/project'

# Create the dataset
root_dir = os.path.join(path_project, 'data/banwuli_data/yukina_data')
# Save as NPY

X_train = np.load(os.path.join(root_dir, 'train/features.npy'))
y_train = np.load(os.path.join(root_dir, 'train/labels.npy'))
ensemeble_data = np.load(os.path.join(root_dir, 'train_seperate/augment', 'no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=100,tau1=0.1', '4', '0.npz'))

X_aug = ensemeble_data['X_train'][np.where(ensemeble_data['y_train'] == 1)]
y_aug = np.ones(X_aug.shape[0])

for fault in tqdm(os.listdir(os.path.join(root_dir, 'test'))):
    path_fault = os.path.join(root_dir, 'test', fault)
    files = os.listdir(path_fault)

    for i in range(1, int(len(files) / 2 + 1)):
        X_test = np.load(os.path.join(root_dir, 'test', fault, f"features_{i}.npy"))
        y_test = np.load(os.path.join(root_dir, 'test', fault, f"labels_{i}.npy"))

        output_path = os.path.join(path_project, root_dir, 'DeepSAD_aug_data', fault)
        os.makedirs(output_path, exist_ok=True)

        np.savez(os.path.join(output_path, f"dataset_{i}.npz"), X=[0], y=[0], X_train=np.concatenate((X_train,X_aug)), y_train=np.concatenate((y_train,y_aug)),
             X_test=X_test, y_test=y_test)

print('Data generate complete.')
