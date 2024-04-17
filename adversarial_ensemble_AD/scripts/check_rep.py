import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
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

path_train = os.path.join(path_project, 'data/banwuli_data/yukina_data', 'normal')
path_test = os.path.join(path_project, 'data/banwuli_data/yukina_data', 'anomaly')

X_train = np.load(os.path.join(path_train, 'features.npy'))
y_train = np.load(os.path.join(path_train, 'labels.npy'))

fault_list = ['ks', 'sf', 'lqs', 'rqs', 'T']
for fault in fault_list:
    path_fault = os.path.join(path_test, fault)
    files = os.listdir(path_fault)
    for i in range(1, int(len(files) / 2 + 1)):
        X_test = np.load(os.path.join(path_fault, f"features_{i}.npy"))
        y_test = np.load(os.path.join(path_fault, f"labels_{i}.npy"))

    X_normal = np.concatenate([X_train, X_test[y_test == 0]], axis=0)
    y_normal = np.concatenate([y_train, y_test[y_test == 0]], axis=0)
    X_anomaly = X_test[y_test == 1]
    y_anomaly = y_test[y_test == 1]

    # 随机抽取等量的正常样本和异常样本
    num_samples = min(len(X_normal), len(X_anomaly))
    normal_indices = np.random.randint(0, len(X_normal), num_samples)
    anomaly_indices = np.random.randint(0, len(X_anomaly), num_samples)

    X_new = np.concatenate((X_normal[normal_indices],
                            X_anomaly[anomaly_indices]))

    y_new = np.concatenate((y_normal[normal_indices],
                            y_anomaly[anomaly_indices]))

    X = X_new
    y = y_new
    # X = np.vstack([X_normal, X_anomaly])
    # y = np.hstack([y_normal, y_anomaly])

    tau = 10
    cluster_num = 2
    # path_plot = os.path.join(path_project,
    #                          f'adversarial_ensemble_AD/log/train_result/before_train/K=2/reps/reps_fault={fault}_tau={tau}')
    dir_plot = os.path.join(path_project, 'adversarial_ensemble_AD/log/train_result/K=2,gan_epoch=100,lam=10,tau=10', 'reps')
    os.makedirs(dir_plot, exist_ok=True)
    path_plot = os.path.join(dir_plot, f'reps_fault={fault}_tau={tau}')
    path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/ensemble/K=2,gan_epoch=100,lam=10,tau=10/4')
    params = {
        "path_detector": path_detector
    }
    ad = Adversarial_Generator(params)
    ad.calculate_reps(X, y, tau=tau, path_plot=path_plot)
