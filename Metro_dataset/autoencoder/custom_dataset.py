import gc
import os

from tqdm import tqdm

path_project = '/home/yukina/Missile_Fault_Detection/project'

import pandas as pd
import numpy as np

from glob import glob

dataset_params = 'window=1, step=1'

dataset_name = 'Metro'

data_path = os.path.join(path_project, f'data/{dataset_name}/yukina_data/DeepSAD_data, {dataset_params}')

output_path = os.path.join(path_project, f'data/{dataset_name}/csv/{dataset_params}')
os.makedirs(output_path, exist_ok=True)  # 创建结果文件夹

train_data = np.load(os.path.join(data_path, 'train.npz'))
X_train = train_data['X_train'].reshape(train_data['X_train'].shape[0], -1)
y_train = train_data['y_train']

test_files = glob(os.path.join(data_path, '*test.npz'))

# 遍历所有数据集文件
for test_path in tqdm(test_files, desc='Total progress'):
    base_name = os.path.basename(test_path).replace('.npz', '')
    # 创建结果文件夹路径

    data = np.load(os.path.join(data_path, test_path))

    X_test = data['X_test'].reshape(data['X_test'].shape[0], -1)
    y_test = data['y_test']

    dataset = {'X': data['X'], 'y': data['y'], 'X_train': X_train, 'y_train': y_train,
               'X_test': X_test, 'y_test': y_test}


    # 假设训练数据和测试数据分别为 train_features, train_labels 和 test_features, test_labels
    # 创建训练数据和测试数据的DataFrame

    test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)


    test_df = pd.DataFrame(test_data)

    # 将DataFrame存储为CSV文件

    test_df.to_csv(os.path.join(output_path, f'{base_name}.csv'))  # 不包含索引和列名

    del data, dataset, test_data, test_df
    gc.collect()

train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
train_df = pd.DataFrame(train_data)
train_df.to_csv(os.path.join(output_path, f'metro-traffic-volume.train.csv'))  # 不包含索引和列名