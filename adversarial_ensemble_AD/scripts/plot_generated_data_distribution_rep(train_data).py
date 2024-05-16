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

iteration = 4
test_set_name = 'banwuli_data'
model_name = 'no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=100,tau1=0.1'
path_plot = os.path.join(path_project, f'adversarial_ensemble_AD/log/{test_set_name}/train_result', model_name, 'generated_data')
os.makedirs(path_plot, exist_ok=True)

path_train_origin = os.path.join(path_project, f'data/banwuli_data/yukina_data/DeepSAD_data/ks/dataset_1.npz')

path_train_new = os.path.join(path_project, f'data/{test_set_name}/yukina_data/train_seperate', 'augment', model_name, f'{iteration}')

path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/DeepSAD_seed=1.pth')
path_detector_aug = os.path.join(path_project, f'adversarial_ensemble_AD/models/banwuli_data/DeepSAD_aug/DeepSAD_seed=1.pth')

random_seed = 42

# 随机抽取的样本数
num_samples = 2000
np.random.seed(0)

train_origin = np.load(path_train_origin)

train_origin = train_origin['X_train'][np.where(train_origin['y_train'] == 0)]
sampled_train_origin = train_origin[np.random.choice(range(0, train_origin.shape[0]), num_samples, replace=True)]

datasets = []
for dataset in os.listdir(path_train_new):
    datasets.append(np.load(os.path.join(path_train_new, dataset)))

generated_data = datasets[0]['X_train'][np.where(datasets[0]['y_train'] == 1)]

test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data')

# plot时是否考虑聚类标签
use_train_cluster_label = False


faults = []
fault_data = []
normal_data = []
for fault in tqdm(os.listdir(test_data_path)):
    base_name = os.path.basename(fault).replace('.npz', '')
    faults.append(base_name)
    # 创建结果文件夹路径

    anomaly_data = []
    path_fault = os.path.join(test_data_path, fault)
    files = os.listdir(path_fault)
    for i in range(1, int(len(files) + 1)):
        data = np.load(os.path.join(path_fault, f"dataset_{i}.npz"))
        dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
                   'X_test': data['X_test'], 'y_test': data['y_test']}
        normal_data.append(dataset['X_test'][np.where(dataset['y_test'] == 0)])
        anomaly_data.append(dataset['X_test'][np.where(dataset['y_test'] == 1)])

    anomaly_data = np.concatenate(anomaly_data)
    sampled_anomaly = anomaly_data[np.random.choice(range(0, anomaly_data.shape[0]), num_samples, replace=True)]
    fault_data.append(sampled_anomaly)

# normal_data = np.concatenate(normal_data)
# sampled_normal = normal_data[np.random.choice(range(0, normal_data.shape[0]), num_samples, replace=False)]

sampled_train_data = []
init_train_data = []
for dataset in datasets:
    train_data = dataset['X_train'][np.where(dataset['y_train'] == 0)]
    init_train_data.append(train_data)
    sampled_train_data.append(train_data[np.random.choice(range(0, train_data.shape[0]), num_samples, replace=True)])

init_train_data = np.concatenate(init_train_data)
sampled_init_train_data = init_train_data[np.random.choice(range(0, init_train_data.shape[0]), num_samples, replace=True)]

# X_plot = np.concatenate((sampled_normal, np.concatenate(fault_data)))
# y_plot = np.concatenate((np.zeros(num_samples), np.concatenate([np.ones(num_samples) * (id + 1) for id, fault in enumerate(fault_data)])))



if use_train_cluster_label:
    # 正常数据的标签大于0，异常数据的标签小于0，生成数据的标签为0
    X_origin = np.concatenate((np.concatenate(sampled_train_data), np.concatenate(fault_data)))
    y_origin = np.concatenate((np.concatenate([np.ones(num_samples) * (id + 1) for id, fault in enumerate(sampled_train_data)]), np.concatenate([np.ones(num_samples) * -(id + 1) for id, fault in enumerate(fault_data)])))
else:
    # 不区分训练数据的聚类标签
    # X_origin = np.concatenate((sampled_init_train_data, np.concatenate(fault_data)))
    X_origin = np.concatenate((sampled_train_origin, np.concatenate(fault_data)))
    y_origin = np.concatenate((np.ones(num_samples), np.concatenate([np.ones(num_samples) * -(id + 1) for id, fault in enumerate(fault_data)])))

sampled_generated = generated_data[np.random.choice(range(0, generated_data.shape[0]), num_samples, replace=True)]

X_train = X_origin
y_train = y_origin

X_all = np.concatenate((X_origin, sampled_generated))
y_all = np.concatenate((y_origin, np.zeros(sampled_generated.shape[0])))


detector1 = DeepSAD(seed=1, load_model=os.path.join(path_detector))
detector1.load_model_from_file()
score, rep_train = detector1.predict_score(X_train)
score, rep_all = detector1.predict_score(X_all)


rep_train = np.array((rep_train))
rep_all = np.array((rep_all))

X_plot = np.concatenate((rep_train, np.array(detector1.deepSAD.c).reshape(1, -1)))
y_plot = y_train

# tsne1 = TSNE(n_components=2, random_state=random_seed)  # n_components表示目标维度
#
# X_2d = tsne1.fit_transform(X_plot)  # 对数据进行降维处理
#
# center = X_2d[-1]
# X_2d = X_2d[:-1]
#
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.figure(figsize=(12, 9))
#
# if use_train_cluster_label:
#     for id, train_data in enumerate(sampled_train_data):
#         plt.scatter(X_2d[y_plot == (id + 1), 0], X_2d[y_plot == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
# else:
#     plt.scatter(X_2d[y_plot == 1, 0], X_2d[y_plot == 1, 1], label=f'正常数据', alpha=0.5)
#
# plt.scatter(X_2d[y_plot == 0, 0], X_2d[y_plot == 0, 1], label='生成数据', alpha=0.5)
#
# for id, fault in enumerate(faults):
#     plt.scatter(X_2d[y_plot == -(id + 1), 0], X_2d[y_plot == -(id + 1), 1], label=fault, alpha=0.5)
#
# plt.scatter(center[0], center[1], c='red', marker='x', label='center')
# plt.legend()
#
# plt.xlim(-150, 150)
# plt.ylim(-150, 150)
#
# plt.title('T-SNE 对真实数据和生成数据的可视化')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.show()
# plt.savefig(os.path.join(path_plot, f'TSNE1 of Generated Data_train_{iteration}.png'))
# plt.close()
# plt.cla()

# # 准备初始化矩阵
# num_all_samples = rep_all.shape[0]
# num_train_samples = rep_train.shape[0]
# embedding_dim = X_2d.shape[1]
#
# # 初始化矩阵，将正常样本的嵌入结果放在前面，其余部分初始化为零（或其他选择）
# init_embeddings = np.zeros((num_all_samples+1, embedding_dim))
# init_embeddings[:num_train_samples] = X_2d
#
# detector2 = DeepSAD(seed=1, load_model=os.path.join(path_detector_aug))
# detector2.load_model_from_file()
# score, rep_train = detector2.predict_score(X_train)
# score, rep_all = detector2.predict_score(X_all)
#
# X_plot = np.concatenate((rep_all, np.array(detector2.deepSAD.c).reshape(1, -1)))
# y_plot = y_all

# tsne2 = TSNE(n_components=2, random_state=random_seed, init=init_embeddings)  # n_components表示目标维度
#
# X_2d = tsne2.fit_transform(X_plot)  # 对数据进行降维处理
#
# center = X_2d[-1]
# X_2d = X_2d[:-1]
#
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.figure(figsize=(12, 9))
#
# if use_train_cluster_label:
#     for id, train_data in enumerate(sampled_train_data):
#         plt.scatter(X_2d[y_plot == (id + 1), 0], X_2d[y_plot == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
# else:
#     plt.scatter(X_2d[y_plot == 1, 0], X_2d[y_plot == 1, 1], label=f'正常数据', alpha=0.5)
#
# plt.scatter(X_2d[y_plot == 0, 0], X_2d[y_plot == 0, 1], label='生成数据', alpha=0.5)
#
# for id, fault in enumerate(faults):
#     plt.scatter(X_2d[y_plot == -(id + 1), 0], X_2d[y_plot == -(id + 1), 1], label=fault, alpha=0.5)
#
# plt.scatter(center[0], center[1], c='red', marker='x', label='center')
# plt.legend()
#
#
# plt.xlim(-150, 150)
# plt.ylim(-150, 150)
#
# plt.title('T-SNE 对真实数据和生成数据的可视化')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.show()
# plt.savefig(os.path.join(path_plot, f'TSNE2 of Generated Data_train_{iteration}.png'))
# plt.close()
# plt.cla()


import umap
umap_model = umap.UMAP(n_components=2, random_state=random_seed)
X_2d = umap_model.fit_transform(X_plot)

center = X_2d[-1]
X_2d = X_2d[:-1]

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(8, 6))

if use_train_cluster_label:
    for id, train_data in enumerate(sampled_train_data):
        plt.scatter(X_2d[y_plot == (id + 1), 0], X_2d[y_plot == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
else:
    plt.scatter(X_2d[y_plot == 1, 0], X_2d[y_plot == 1, 1], label=f'正常数据', alpha=0.5)

plt.scatter(X_2d[y_plot == 0, 0], X_2d[y_plot == 0, 1], label='生成数据', alpha=0.5)

for id, fault in enumerate(faults):
    plt.scatter(X_2d[y_plot == -(id + 1), 0], X_2d[y_plot == -(id + 1), 1], label=fault, alpha=0.5)

plt.scatter(center[0], center[1], c='red', marker='x', label='center')
plt.legend()



plt.title('T-SNE 对真实数据和生成数据的可视化')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.savefig(os.path.join(path_plot, f'UMAP of Generated Data_train_{iteration}.png'))
plt.close()