import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from adbench_modified.baseline.DeepSAD.src.run import DeepSAD
from adversarial_ensemble_AD.data_generate.gan import Adversarial_Generator

path_project = '/home/yukina/Missile_Fault_Detection/project'

path_train = os.path.join(path_project, 'data/banwuli_data/yukina_data', 'train')
path_test = os.path.join(path_project, 'data/banwuli_data/yukina_data', 'test')

X_train = np.load(os.path.join(path_train, 'features.npy'))
y_train = np.load(os.path.join(path_train, 'labels.npy'))

fault_list = ['sf']
for fault in fault_list:
    path_fault = os.path.join(path_test, fault)
    files = os.listdir(path_fault)

    X_test = []
    y_test = []
    for i in range(1, int(len(files) / 2 + 1)):
        X = np.load(os.path.join(path_fault, f"features_{i}.npy"))
        y = np.load(os.path.join(path_fault, f"labels_{i}.npy"))
        X_test.append(X)
        y_test.append(y)

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    X_normal = np.concatenate([X_train, X_test[y_test == 0]], axis=0)
    y_normal = np.concatenate([y_train, y_test[y_test == 0]], axis=0)
    X_anomaly = X_test[y_test == 1]
    y_anomaly = y_test[y_test == 1]

    # 随机抽取等量的正常样本和异常样本
    num_samples = 2000
    # X = np.vstack([X_normal, X_anomaly])
    # y = np.hstack([y_normal, y_anomaly])

    tau = 10
    cluster_num = 7
    iteration = 4
    model_name = 'right_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=0.1,tau1=10,tau2=0.001'
    # path_plot = os.path.join(path_project,
    #                          f'adversarial_ensemble_AD/log/train_result/before_train/K=2/reps/reps_fault={fault}_tau={tau}')
    dir_plot = os.path.join(path_project, 'adversarial_ensemble_AD/log/real_data/train_result', model_name, 'reps')
    os.makedirs(dir_plot, exist_ok=True)
    path_plot = os.path.join(dir_plot, f'reps_train_fault={fault}_tau={tau}')
    path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/real_data/ensemble', model_name, f'{iteration}')

    path_train_new = os.path.join(path_project, 'data/real_data/yukina_data/train_seperate', 'augment', model_name, f'{iteration}')

    train_datasets = []
    for dataset in os.listdir(path_train_new):
        train_datasets.append(np.load(os.path.join(path_train_new, dataset)))

    sampled_train_data = []

    for dataset in train_datasets:
        train_data = dataset['X_train'][np.where(dataset['y_train'] == 0)]
        sampled_train_data.append(
            train_data[np.random.choice(range(0, train_data.shape[0]), num_samples, replace=True)])

    normal_indices = np.random.randint(0, len(X_normal), num_samples)
    anomaly_indices = np.random.randint(0, len(X_anomaly), num_samples)

    # X_new = np.concatenate((X_normal[normal_indices],
    #                         X_anomaly[anomaly_indices]))
    #
    # y_new = np.concatenate((y_normal[normal_indices],
    #                         y_anomaly[anomaly_indices]))

    detectors = []
    scores = []
    for num, model in enumerate(os.listdir(path_detector)):
        X_new = np.concatenate((sampled_train_data[num],
                                X_anomaly[anomaly_indices]))

        y_new = np.concatenate((np.zeros(num_samples),
                                y_anomaly[anomaly_indices]))

        detector = DeepSAD(seed=0, load_model=os.path.join(path_detector, model))
        detector.load_model_from_file()
        detectors.append(detector)
        score, outputs = detector.predict_score(X_new)

        X_plot = np.concatenate((np.array(outputs), np.array(detector.deepSAD.c).reshape(1, -1)))
        tsne = TSNE(n_components=2, random_state=42, perplexity=20)  # n_components表示目标维度

        X_2d = tsne.fit_transform(X_plot)  # 对数据进行降维处理

        center = X_2d[-1]
        X_2d = X_2d[:-1]

        plt.figure(figsize=(8, 6))
        if y_new is not None:
            # 如果有目标数组，根据不同的类别用不同的颜色绘制
            for i in np.unique(y_new):
                plt.scatter(X_2d[y_new == i, 0], X_2d[y_new == i, 1], label=i, alpha=0.5)
            plt.legend()
        else:
            # 如果没有目标数组，直接绘制
            plt.scatter(X_2d[:, 0], X_2d[:, 1])
        plt.scatter(center[0], center[1], c='red', marker='x', label='center')
        plt.title('t-SNE Visualization of rep after training')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
        plt.savefig(path_plot + f'_model={num}.png')
        plt.close()

