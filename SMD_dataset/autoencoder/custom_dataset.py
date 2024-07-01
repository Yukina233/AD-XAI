import os

from tqdm import tqdm

path_project = '/home/yukina/Missile_Fault_Detection/project'

import pandas as pd
import numpy as np

dataset_name = 'window=100, step=10'
data_path = os.path.join(path_project, f'data/SMD/yukina_data/DeepSAD_data, {dataset_name}')

output_path = os.path.join(path_project, f'data/SMD/csv/{dataset_name}')
os.makedirs(output_path, exist_ok=True)  # 创建结果文件夹

# 遍历所有数据集文件
for dataset_path in tqdm(os.listdir(data_path), desc='Total progress'):
    base_name = os.path.basename(dataset_path).replace('.npz', '')
    # 创建结果文件夹路径

    data = np.load(os.path.join(data_path, dataset_path))
    dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
               'X_test': data['X_test'], 'y_test': data['y_test']}


    # 假设训练数据和测试数据分别为 train_features, train_labels 和 test_features, test_labels
    # 创建训练数据和测试数据的DataFrame
    train_data = np.concatenate((dataset['X_train'].reshape(dataset['X_train'].shape[0], -1), dataset['y_train'].reshape(-1, 1)), axis=1)
    test_data = np.concatenate((dataset['X_test'].reshape(dataset['X_test'].shape[0], -1), dataset['y_test'].reshape(-1, 1)), axis=1)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # 将DataFrame存储为CSV文件
    train_df.to_csv(os.path.join(output_path, f'{base_name}.train.csv'))  # 不包含索引和列名
    test_df.to_csv(os.path.join(output_path, f'{base_name}.test.csv'))  # 不包含索引和列名
