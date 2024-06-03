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

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

iteration = 0
test_set_name = 'SMD'
model_name_template = 'std, window=100, step=10, no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2={},tau1=1'
output_dir_template = os.path.join(path_project, f'SMD_dataset/log/{test_set_name}/train_result')

train_new_dir_template = os.path.join(path_project,
                                      f'data/{test_set_name}/yukina_data/ensemble_data, window=100, step=10', 'augment')

detector_dir_template = os.path.join(path_project, f'SMD_dataset/models/{test_set_name}/ensemble')

test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data, window=100, step=10')

lam2_values = [0.1, 1, 10, 100]  # 修改不同的lam2参数值

for dataset_name in os.listdir(os.path.join(train_new_dir_template, model_name_template.format(lam2_values[0]))):
    path_plot = os.path.join(output_dir_template, dataset_name, 'generated_data')
    os.makedirs(path_plot, exist_ok=True)

    # 读取测试数据
    data = np.load(os.path.join(test_data_path, f"{dataset_name}.npz"))
    dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
               'X_test': data['X_test'], 'y_test': data['y_test']}
    anomaly_data = dataset['X_test'][np.where(dataset['y_test'] == 1)]
    normal_data = dataset['X_train'][np.where(dataset['y_train'] == 0)]

    np.random.seed(0)
    num_samples = 2000
    sampled_anomaly = anomaly_data[np.random.choice(range(0, anomaly_data.shape[0]), num_samples, replace=True)]
    sampled_normal = normal_data[np.random.choice(range(0, normal_data.shape[0]), num_samples, replace=True)]

    plt.figure(figsize=(12, 9))
    plt.rcParams['font.sans-serif'] = ['SimSun']

    X_plot = sampled_normal
    y_plot = np.zeros(num_samples)

    for lam2 in lam2_values:
        model_name = model_name_template.format(lam2)
        train_new_dir = os.path.join(train_new_dir_template, model_name)

        path_train_new = os.path.join(train_new_dir, dataset_name, f'{iteration}')

        datasets = []
        for dataset in os.listdir(path_train_new):
            datasets.append(np.load(os.path.join(path_train_new, dataset)))

        generated_data = datasets[0]['X_train'][np.where(datasets[0]['y_train'] == 1)]
        sampled_generated = generated_data[
            np.random.choice(range(0, generated_data.shape[0]), num_samples, replace=True)]

        X_plot = np.concatenate((X_plot, sampled_generated))
        y_plot = np.concatenate((y_plot, np.zeros(num_samples) - lam2))  # 用负的lam2值作为标签区分

    X_plot = np.concatenate((X_plot, sampled_anomaly))
    y_plot = np.concatenate((y_plot, np.ones(num_samples)))

    tsne = TSNE(n_components=2, random_state=0)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(X_plot)
    X_2d = tsne.fit_transform(normalized_data)

    plt.scatter(X_2d[y_plot == 0, 0], X_2d[y_plot == 0, 1], label='正常数据', alpha=0.5)
    plt.scatter(X_2d[y_plot == 1, 0], X_2d[y_plot == 1, 1], label='故障数据', alpha=0.5)

    for lam2 in lam2_values:
        plt.scatter(X_2d[y_plot == -lam2, 0], X_2d[y_plot == -lam2, 1], label=f'生成数据 lam2={lam2}', alpha=0.5)

    plt.legend()
    plt.title(f'T-SNE 对真实数据和生成数据的可视化 - {dataset_name}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(os.path.join(path_plot, f'TSNE_of_Generated_Data_{dataset_name}_iteration_{iteration}.png'))
    plt.show()
    plt.close()
