import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


# This file is used to genrate dataset for each AD task, with iot data.





    # # Save as NPZ
    # output_path = path_project + f"/data/iot_data_baseline"
    # if not os.path.isdir(output_path):
    #     os.mkdir(output_path)
    # np.savez(os.path.join(output_path, f"iot_{id}.npz"), X=X, y=y, X_train=X_train, y_train=y_train,
    #          X_test=X_test, y_test=y_test)


path_project = '/home/yukina/Missile_Fault_Detection/project'

# Create the dataset
root_dir = path_project + '/data/iot_data'

data_all = []
files = os.listdir(root_dir)
for file in files:
    data_all.append(np.load(os.path.join(root_dir, file), allow_pickle=True))

datasets_num = len(data_all)
# 生成所有可能的8个数据集的组合
combinations_of_datasets = list(combinations(range(datasets_num), datasets_num - 1))

# 对于每个组合，合并8个数据集为新的训练集，剩下的一个为测试集
for combo in tqdm(combinations_of_datasets):
    # 初始化新的训练集和测试集的特征和标签
    X_train_combined = []
    y_train_combined = []

    # 确定测试集的索引（9个数据集中没有被选为训练集的那个）
    test_index = list(set(range(datasets_num)) - set(combo))[0]
    X_test = data_all[test_index]['X_test']
    y_test = data_all[test_index]['y_test']

    # 合并训练集
    for i in combo:
        X_train_combined.append(data_all[i]['X_train'])
        y_train_combined.append(data_all[i]['y_train'])

    # 将训练集的数据合并成一个大的数据集
    X_train_combined = np.concatenate(X_train_combined, axis=0)
    y_train_combined = np.concatenate(y_train_combined, axis=0)

    # Save as NPZ
    output_path = path_project + f"/data/iot_data_baseline"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    np.savez(os.path.join(output_path, f"iot_{test_index}.npz"), X=[0], y=[0], X_train=X_train_combined, y_train=y_train_combined,
             X_test=X_test, y_test=y_test)

print('Data generate complete.')
