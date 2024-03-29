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

X_train = np.load(os.path.join(root_dir, 'normal/features.npy'))
y_train = np.load(os.path.join(root_dir, 'normal/labels.npy'))

for fault in tqdm(os.listdir(os.path.join(root_dir, 'anomaly'))):
    path_fault = os.path.join(root_dir, 'anomaly', fault)
    files = os.listdir(path_fault)

    for i in range(1, int(len(files) / 2 + 1)):
        X_test = np.load(os.path.join(root_dir, 'anomaly', fault, f"features_{i}.npy"))
        y_test = np.load(os.path.join(root_dir, 'anomaly', fault, f"labels_{i}.npy"))

        output_path = os.path.join(path_project, root_dir, 'DeepSAD_data', fault)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        np.savez(os.path.join(output_path, f"dataset_{i}.npz"), X=[0], y=[0], X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)

print('Data generate complete.')
