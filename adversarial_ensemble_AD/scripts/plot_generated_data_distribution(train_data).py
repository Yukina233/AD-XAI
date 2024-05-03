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
test_set_name = 'real_data'
model_name = 'right_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=40,tau1=0.1,tau2=1'
path_plot = os.path.join(path_project, f'adversarial_ensemble_AD/log/{test_set_name}/train_result', model_name, 'generated_data')
os.makedirs(path_plot, exist_ok=True)

path_train_new = os.path.join(path_project, f'data/{test_set_name}/yukina_data/train_seperate', 'augment', model_name, f'{iteration}')

path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/{test_set_name}/ensemble', model_name, f'{iteration}')

datasets = []
for dataset in os.listdir(path_train_new):
    datasets.append(np.load(os.path.join(path_train_new, dataset)))

generated_data = datasets[0]['X_train'][np.where(datasets[0]['y_train'] == 1)]

test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data')

# plot时是否考虑聚类标签
use_train_cluster_label = False
# 随机抽取的样本数
num_samples = 2000
np.random.seed(0)

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

# detectors = []
# for num, model in enumerate(os.listdir(path_detector)):
#     detector = DeepSAD(seed=0, load_model=os.path.join(path_detector, model))
#     detector.load_model_from_file()
#     detectors.append(detector)
#     score, outputs = detector.predict_score(X)
#
#     X_plot = np.concatenate((np.array(outputs), np.array(detector.deepSAD.c).reshape(1, -1)))
#     tsne = TSNE(n_components=2, random_state=42)  # n_components表示目标维度
#
#     X_2d = tsne.fit_transform(X_plot)  # 对数据进行降维处理
#
#     center = X_2d[-1]
#     X_2d = X_2d[:-1]

if use_train_cluster_label:
    # 正常数据的标签大于0，异常数据的标签小于0，生成数据的标签为0
    X_plot = np.concatenate((np.concatenate(sampled_train_data), np.concatenate(fault_data)))
    y_plot = np.concatenate((np.concatenate([np.ones(num_samples) * (id + 1) for id, fault in enumerate(sampled_train_data)]), np.concatenate([np.ones(num_samples) * -(id + 1) for id, fault in enumerate(fault_data)])))
else:
    # 不区分训练数据的聚类标签
    X_plot = np.concatenate((sampled_init_train_data, np.concatenate(fault_data)))
    y_plot = np.concatenate((np.ones(num_samples), np.concatenate([np.ones(num_samples) * -(id + 1) for id, fault in enumerate(fault_data)])))

sampled_generated = generated_data[np.random.choice(range(0, generated_data.shape[0]), num_samples, replace=True)]
X_plot = np.concatenate((X_plot, sampled_generated))
y_plot = np.concatenate((y_plot, np.zeros(sampled_generated.shape[0])))


tsne = TSNE(n_components=2, random_state=0)  # n_components表示目标维度

X_2d = tsne.fit_transform(X_plot)  # 对数据进行降维处理

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(12, 9))

if use_train_cluster_label:
    for id, train_data in enumerate(sampled_train_data):
        plt.scatter(X_2d[y_plot == (id + 1), 0], X_2d[y_plot == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
else:
    plt.scatter(X_2d[y_plot == 1, 0], X_2d[y_plot == 1, 1], label=f'正常数据', alpha=0.5)

plt.scatter(X_2d[y_plot == 0, 0], X_2d[y_plot == 0, 1], label='生成数据', alpha=0.5)

for id, fault in enumerate(faults):
    plt.scatter(X_2d[y_plot == -(id + 1), 0], X_2d[y_plot == -(id + 1), 1], label=fault, alpha=0.5)

plt.legend()

plt.title('T-SNE 对真实数据和生成数据的可视化')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.savefig(os.path.join(path_plot, f'TSNE of Generated Data_train_{iteration}.png'))
plt.close()
