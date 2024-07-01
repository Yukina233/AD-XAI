import argparse
import pickle
import random

import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
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

iteration = 0
test_set_name = 'Daphnet'

output_dir = os.path.join(path_project, f'Daphnet_dataset/log/{test_set_name}/train_result', 'collection')


train_data_path = os.path.join(path_project,
                               f'data/{test_set_name}/yukina_data/DeepSAD_data, window=100, step=10/train.npz')
test_data_path = os.path.join(path_project,
                              f'data/{test_set_name}/yukina_data/DeepSAD_data, window=100, step=10/test.npz')

path_plot = os.path.join(output_dir, 'generated_data')
os.makedirs(path_plot, exist_ok=True)

model_name1 = 'no_GAN, std, window=100, step=10, no_tau2_K=7,deepsad_epoch=50,gan_epoch=20,lam1=0.1,lam2=0.9,tau1=1'
train_new_dir1 = os.path.join(path_project, f'data/{test_set_name}/yukina_data/ensemble_data, window=100, step=10',
                              'augment', model_name1)

path_train_new1 = os.path.join(train_new_dir1, f'{iteration}')
aug_datasets = []
for dataset in os.listdir(path_train_new1):
    aug_datasets.append(np.load(os.path.join(path_train_new1, dataset)))

generated_data = []
for dataset in aug_datasets:
    generated_data.append(dataset['X_train'][np.where(dataset['y_train'] == 1)])
generated_data1 = np.concatenate(generated_data)

model_name2 = 'no_GAN, std, window=100, step=10, no_tau2_K=7,deepsad_epoch=50,gan_epoch=20,lam1=0.8,lam2=0.2,tau1=1'
train_new_dir2 = os.path.join(path_project, f'data/{test_set_name}/yukina_data/ensemble_data, window=100, step=10',
                              'augment', model_name2)

path_train_new2 = os.path.join(train_new_dir2, f'{iteration}')
aug_datasets = []
for dataset in os.listdir(path_train_new2):
    aug_datasets.append(np.load(os.path.join(path_train_new2, dataset)))

generated_data = []
for dataset in aug_datasets:
    generated_data.append(dataset['X_train'][np.where(dataset['y_train'] == 1)])
generated_data2 = np.concatenate(generated_data)

model_name3 = 'no_GAN, std, window=100, step=10, no_tau2_K=7,deepsad_epoch=50,gan_epoch=20,lam1=0.9,lam2=0.1,tau1=1'
train_new_dir3 = os.path.join(path_project, f'data/{test_set_name}/yukina_data/ensemble_data, window=100, step=10',
                              'augment', model_name3)

path_train_new3 = os.path.join(train_new_dir3, f'{iteration}')
aug_datasets = []
for dataset in os.listdir(path_train_new3):
    aug_datasets.append(np.load(os.path.join(path_train_new3, dataset)))

generated_data = []
for dataset in aug_datasets:
    generated_data.append(dataset['X_train'][np.where(dataset['y_train'] == 1)])
generated_data3 = np.concatenate(generated_data)

# plot时是否考虑聚类标签
use_train_cluster_label = False
# 随机抽取的样本数
num_samples = 5000
np.random.seed(0)

data = np.load(os.path.join(test_data_path))
dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
           'X_test': data['X_test'], 'y_test': data['y_test']}
anomaly_data = dataset['X_test'][np.where(dataset['y_test'] == 1)]
sampled_anomaly = anomaly_data[np.random.choice(range(0, anomaly_data.shape[0]), num_samples, replace=True)]

sampled_train_data = []
init_train_data = []

data = np.load(os.path.join(train_data_path))
train_data = data['X_train'][np.where(data['y_train'] == 0)]
init_train_data.append(train_data)
sampled_train_data.append(train_data[np.random.choice(range(0, train_data.shape[0]), num_samples, replace=True)])

init_train_data = np.concatenate(init_train_data)
sampled_init_train_data = init_train_data[
    np.random.choice(range(0, init_train_data.shape[0]), num_samples, replace=True)]

if use_train_cluster_label:
    assert NotImplementedError
    # 正常数据的标签大于0，异常数据的标签小于0，生成数据的标签为0
    X_plot = np.concatenate((np.concatenate(sampled_train_data), sampled_anomaly))
    y_plot = np.concatenate((np.concatenate(
        [np.ones(num_samples) * (id + 1) for id, fault in enumerate(sampled_train_data)]), -np.ones(num_samples)))
else:
    # 不区分训练数据的聚类标签
    X_plot = np.concatenate((sampled_init_train_data, sampled_anomaly))
    y_plot = np.concatenate((np.zeros(num_samples), -np.ones(num_samples)))

sampled_generated1 = generated_data1[np.random.choice(range(0, generated_data1.shape[0]), num_samples, replace=True)]
sampled_generated2 = generated_data2[np.random.choice(range(0, generated_data2.shape[0]), num_samples, replace=True)]
sampled_generated3 = generated_data3[np.random.choice(range(0, generated_data2.shape[0]), num_samples, replace=True)]

X_train = X_plot
y_train = y_plot

X_all = np.concatenate((X_plot, sampled_generated1, sampled_generated2, sampled_generated3))
y_all = np.concatenate((y_plot, np.ones(sampled_generated1.shape[0]), 2 * np.ones(sampled_generated2.shape[0]), 3 * np.ones(sampled_generated3.shape[0])))

# import umap
# umap_model = umap.UMAP(n_components=2, random_state=42)
# X_2d = umap_model.fit_transform(X_train)

tsne1 = TSNE(n_components=2, random_state=0)  # n_components表示目标维度

# 创建MinMaxScaler对象
scaler1 = MinMaxScaler()
# 对数据进行归一化
normalized_data = scaler1.fit_transform(X_all)
X_2d = tsne1.fit_transform(normalized_data)  # 对数据进行降维处理

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(12, 9))

if use_train_cluster_label:
    for id, train_data in enumerate(sampled_train_data):
        plt.scatter(X_2d[y_all == (id + 1), 0], X_2d[y_all == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
else:
    plt.scatter(X_2d[y_all == 0, 0], X_2d[y_all == 0, 1], label=f'正常数据', alpha=0.5)

plt.scatter(X_2d[y_all == 1, 0], X_2d[y_all == 1, 1], label='生成数据(no_GAN,lam=0.1)', alpha=0.5)
plt.scatter(X_2d[y_all == 2, 0], X_2d[y_all == 2, 1], label='生成数据(no_GAN,lam=0.8)', alpha=0.5)
plt.scatter(X_2d[y_all == 3, 0], X_2d[y_all == 3, 1], label='生成数据(no_GAN,lam=0.9)', alpha=0.5)

plt.scatter(X_2d[y_all == -1, 0], X_2d[y_all == -1, 1], label='故障数据', alpha=0.5)
# for id, fault in enumerate(faults):
#     plt.scatter(X_2d[y_train == -(id + 1), 0], X_2d[y_train == -(id + 1), 1], label=fault, alpha=0.5)

# plt.xlim([-25, 25])
# plt.ylim([-25, 25])
plt.legend()

plt.title('T-SNE 对真实数据和生成数据的可视化')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.savefig(os.path.join(path_plot, f'TSNE1 of Generated Data_train_{iteration}.png'))
plt.close()

plt.cla()

# # 准备初始化矩阵
# num_all_samples = X_all.shape[0]
# num_train_samples = X_train.shape[0]
# embedding_dim = X_2d.shape[1]
#
# # 初始化矩阵，将正常样本的嵌入结果放在前面，其余部分初始化为零（或其他选择）
# init_embeddings = np.zeros((num_all_samples, embedding_dim))
# init_embeddings[:num_train_samples] = X_2d


# import umap
# umap_model = umap.UMAP(n_components=2, random_state=42)
# X_2d = umap_model.fit_transform(X_all)
# tsne2 = TSNE(n_components=2, random_state=0)  # n_components表示目标维度
# # 创建MinMaxScaler对象
# scaler1 = MinMaxScaler()
# # 对数据进行归一化
# normalized_data = scaler1.fit_transform(X_all)
# X_2d = tsne2.fit_transform(normalized_data)  # 对数据进行降维处理
#
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.figure(figsize=(12, 9))
#
# if use_train_cluster_label:
#     for id, train_data in enumerate(sampled_train_data):
#         plt.scatter(X_2d[y_all == (id + 1), 0], X_2d[y_all == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
# else:
#     plt.scatter(X_2d[y_all == 1, 0], X_2d[y_all == 1, 1], label=f'正常数据', alpha=0.5)
#
# plt.scatter(X_2d[y_all == 0, 0], X_2d[y_all == 0, 1], label='生成数据', alpha=0.5)
#
# plt.scatter(X_2d[y_all == -1, 0], X_2d[y_all == -1, 1], label='故障数据', alpha=0.5)
# # for id, fault in enumerate(faults):
# #     plt.scatter(X_2d[y_all == -(id + 1), 0], X_2d[y_all == -(id + 1), 1], label=fault, alpha=0.5)
#
# # plt.xlim([-25, 25])
# # plt.ylim([-25, 25])
# plt.legend()
#
# plt.title('T-SNE 对真实数据和生成数据的可视化')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.show()
# plt.savefig(os.path.join(path_plot, f'TSNE2 of Generated Data_train_{iteration}.png'))
# plt.close()
