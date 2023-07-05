import glob
import os
import pickle
import time

import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

path_project = '/home/yukina/Missile_Fault_Detection/project/'
sub_path = 'IF/seed=0/group_reduce_pos/'


def load_all_pickle_files(directory):
    # 使用glob找到目录中所有的.h5文件
    pickle_files = glob.glob(os.path.join(directory, '*.pkl'))

    # 加载所有的模型并存储在一个字典中
    files = {}
    for file in pickle_files:
        file_name = os.path.basename(file).split('.')[0]
        files[file_name] = pickle.load(open(file, 'rb'))

    return files


data_path = path_project + sub_path
filename_format_pos = "model-remove-{percentage}%-train_seed={train_seed}-{sign}-history.pkl"
sign = 'pos'
percentages = [5, 10, 15, 20, 25]  # 示例的百分比值列表
train_seeds = [0, 1, 2]  # 对于每个percentage的种子列表
data = []
for percentage in percentages:
    percentage_data = []

    for seed in train_seeds:
        filename = filename_format_pos.format(percentage=percentage, train_seed=seed, sign=sign)
        filepath = data_path + filename
        with open(filepath, 'rb') as file:
            seed_data = pickle.load(file)
            seed_data['loss'] = np.expand_dims(np.array(seed_data['loss']), axis=0)
            seed_data['accuracy'] = np.expand_dims(np.array(seed_data['accuracy']), axis=0)
            seed_data['val_loss'] = np.expand_dims(np.array(seed_data['val_loss']), axis=0)
            seed_data['val_accuracy'] = np.expand_dims(np.array(seed_data['val_accuracy']), axis=0)
            percentage_data.append(seed_data)
    data.append(percentage_data)

# 合并不同seed的data
all_pickle_files = {}
for i, percentage in enumerate(percentages):
    all_pickle_files[f'remove-{percentage}%-{sign}-group'] = {}
    for key in data[0][0].keys():
        all_pickle_files[f'remove-{percentage}%-{sign}-group'][key] = np.concatenate([data[i][j][key] for j in range(3)])

    # 计算均值和方差
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['mean_loss'] = np.mean(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['loss'], axis=0)
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['mean_accuracy'] = np.mean(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['accuracy'], axis=0)
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['mean_val_loss'] = np.mean(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['val_loss'], axis=0)
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['mean_val_accuracy'] = np.mean(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['val_accuracy'], axis=0)
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['std_loss'] = np.std(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['loss'],
        axis=0)
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['std_accuracy'] = np.std(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['accuracy'], axis=0)
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['std_val_loss'] = np.std(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['val_loss'], axis=0)
    all_pickle_files[f'remove-{percentage}%-{sign}-group']['std_val_accuracy'] = np.std(
        all_pickle_files[f'remove-{percentage}%-{sign}-group']['val_accuracy'], axis=0)

# 输出到pkl文件
for key, value in all_pickle_files.items():
    with open(data_path + f'{key}-train_seed={train_seeds.__str__()}.pkl', 'wb') as file:
        pickle.dump(value, file)
print('Group Done!')

# # 绘制验证集的损失函数
# plt.figure(figsize=(10, 10))
# for file_name, history in all_pickle_files.items():
#     plt.plot(history['val_loss'], label=file_name)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Validation Loss')
# plt.legend()
# plt.show()
#
# # 绘制验证集的准确率
# plt.figure(figsize=(10, 10))
# for file_name, history in all_pickle_files.items():
#     plt.plot(history['val_accuracy'], label=file_name)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Validation Accuracy')
# plt.legend()
# plt.show()
