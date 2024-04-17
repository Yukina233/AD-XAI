import argparse
import pickle
import random

import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import glob

from adversarial_ensemble_AD.data_generate.gan import Adversarial_Generator

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

model_name = 'K=2,gan_epoch=100,lam=10,tau=10'
path_plot = os.path.join(path_project, 'adversarial_ensemble_AD/log/train_result', model_name, 'generated_data')
os.makedirs(path_plot, exist_ok=True)

path_train_new = os.path.join(path_project, 'data/banwuli_data/yukina_data/train_seperate', 'augment', model_name, '4')

new_dataset = np.load(os.path.join(path_train_new, '0.npz'))
generated_data = new_dataset['X_train'][np.where(new_dataset['y_train'] == 1)]

test_data_path = os.path.join(path_project, 'data/banwuli_data/yukina_data/DeepSAD_data')

# 随机抽取的样本数
num_samples = 1000
np.random.seed(0)

faults = []
fault_data = []
normal_data = []
# for fault in tqdm(os.listdir(test_data_path)):
#     base_name = os.path.basename(fault).replace('.npz', '')
#     faults.append(base_name)
#     # 创建结果文件夹路径
#
#     anomaly_data = []
#     path_fault = os.path.join(test_data_path, fault)
#     files = os.listdir(path_fault)
#     for i in range(1, int(len(files) + 1)):
#         data = np.load(os.path.join(path_fault, f"dataset_{i}.npz"))
#         dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
#                    'X_test': data['X_test'], 'y_test': data['y_test']}
#         normal_data.append(dataset['X_test'][np.where(dataset['y_test'] == 0)])
#         anomaly_data.append(dataset['X_test'][np.where(dataset['y_test'] == 1)])
#
#     anomaly_data = np.concatenate(anomaly_data)
#     sampled_anomaly = anomaly_data[np.random.choice(range(0, anomaly_data.shape[0]), num_samples, replace=False)]
#     fault_data.append(sampled_anomaly)

normal_data = np.concatenate(normal_data)

sampled_normal = normal_data[np.random.choice(range(0, normal_data.shape[0]), num_samples, replace=False)]

X_plot = np.concatenate((sampled_normal, np.concatenate(fault_data)))
y_plot = np.concatenate((np.zeros(num_samples), np.concatenate([np.ones(num_samples) * (id + 1) for id, fault in enumerate(fault_data)])))

sampled_generated = generated_data[np.random.choice(range(0, generated_data.shape[0]), num_samples, replace=False)]
X_plot = np.concatenate((X_plot, sampled_generated))
y_plot = np.concatenate((y_plot, np.ones(sampled_generated.shape[0]) * -1))


tsne = TSNE(n_components=2, random_state=0)  # n_components表示目标维度

X_2d = tsne.fit_transform(X_plot)  # 对数据进行降维处理

plt.figure(figsize=(16, 12))


plt.scatter(X_2d[y_plot == 0, 0], X_2d[y_plot == 0, 1], label='normal', alpha=0.5)
plt.scatter(X_2d[y_plot == -1, 0], X_2d[y_plot == -1, 1], label='generated', alpha=0.5)

# for id, fault in enumerate(faults):
#     plt.scatter(X_2d[y_plot == (id + 1), 0], X_2d[y_plot == (id + 1), 1], label=fault, alpha=0.5)

plt.legend()

plt.title('t-SNE Visualization of generated data and real data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.savefig(os.path.join(path_plot, f'TSNE of Generated Data.png'))
plt.close()
