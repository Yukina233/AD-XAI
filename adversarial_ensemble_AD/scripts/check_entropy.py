import os
import numpy as np
import torch

from matplotlib import pyplot as plt

from adversarial_ensemble_AD.data_generate.gan import Adversarial_Generator

path_project = '/home/yukina/Missile_Fault_Detection/project'


def check_test_data_entropy():
    test_set_name = 'banwuli_data'
    path_train = os.path.join(path_project, f'data/{test_set_name}/yukina_data', 'train')
    path_test = os.path.join(path_project, f'data/{test_set_name}/yukina_data', 'test')

    X_train = np.load(os.path.join(path_train, 'features.npy'))
    y_train = np.load(os.path.join(path_train, 'labels.npy'))

    fault_list = ['ks', 'sf', 'lqs', 'rqs', 'T']
    # fault_list = ['sf']
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
        num_samples = 1000
        normal_indices = np.random.randint(0, len(X_normal), num_samples)
        anomaly_indices = np.random.randint(0, len(X_anomaly), num_samples)

        X_new = np.concatenate((X_normal[normal_indices],
                                X_anomaly[anomaly_indices]))

        y_new = np.concatenate((y_normal[normal_indices],
                                y_anomaly[anomaly_indices]))

        X = X_new
        y = y_new

        tau = 0.1
        iteration = 4
        cluster_num = 7

        model_name = 'no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=1,tau1=0.1'
        dir_plot = os.path.join(path_project, f'adversarial_ensemble_AD/log/{test_set_name}/train_result/{model_name}/entropys')
        os.makedirs(dir_plot, exist_ok=True)
        path_plot = os.path.join(path_project, dir_plot, f'after_train_{iteration}_fault={fault}_tau={tau}.png')
        path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/{test_set_name}/ensemble/{model_name}/{iteration}')
        params = {
            "path_detector": path_detector
        }
        ad = Adversarial_Generator(params)
        X = torch.tensor(X, device='cuda', dtype=torch.float32)

        entropys = ad.calculate_entropy_test(X, tau=tau).cpu().detach().numpy()
        # entropys = ad.calculate_entropy_numpy(X, tau=tau)

        entropys = np.nan_to_num(entropys, 0)

        if path_plot is not None:
            # 按照标签绘制直方图
            plt.cla()
            plt.rcParams['font.sans-serif'] = ['SimSun']
            # plt.figure(figsize=(8, 6))
            plt.hist(x=entropys[y == 0], bins=50, alpha=0.5, label='正常', color='blue', range=(0, 2))
            plt.hist(x=entropys[y == 1], bins=50, alpha=0.5, label='故障', color='red', range=(0, 2))

            if fault == 'ks':
                fault_CN = '舵面卡死'
            elif fault == 'sf':
                fault_CN = '舵面松浮'
            elif fault == 'lqs':
                fault_CN = '升力面缺损'
            elif fault == 'rqs':
                fault_CN = '舵面缺损'
            elif fault == 'T':
                fault_CN = '推力损失'
            else:
                fault_CN = '故障'

            plt.title(f"正常样本和{fault_CN}样本的熵的频率分布")
            plt.xlabel("熵")
            plt.ylabel("频率")
            plt.legend()
            # plt.tight_layout()
            plt.show()
            plt.savefig(path_plot)


def check_entropy_aug():
    iteration = 0
    test_set_name = 'banwuli_data'
    model_name = 'no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=1,tau1=0.1'
    path_data_check = os.path.join(path_project,
                                   f'data/{test_set_name}/yukina_data/train_seperate/augment/{model_name}/{iteration}')
    path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/{test_set_name}/ensemble/{model_name}/{iteration}')

    tau = 0.1
    cluster_num = 7

    path_plot = os.path.join(path_project,
                             f'adversarial_ensemble_AD/log/{test_set_name}/train_result/{model_name}', 'entropys')
    os.makedirs(path_plot, exist_ok=True)
    path_plot = os.path.join(path_plot, f'{iteration}.png')
    params = {
        "path_detector": path_detector
    }
    ad = Adversarial_Generator(params)

    for dataset in os.listdir(path_data_check):
        data = np.load(os.path.join(path_data_check, dataset))
        X_train = data['X_train']
        y_train = data['y_train']

        X_normal = X_train[y_train == 0]
        y_normal = y_train[y_train == 0]

        X_anomaly = X_train[y_train == 1]
        y_anomaly = y_train[y_train == 1]

        # 随机抽取等量的正常样本和异常样本
        num_samples = 1000
        normal_indices = np.random.randint(0, len(X_normal), num_samples)
        anomaly_indices = np.random.randint(0, len(X_anomaly), num_samples)

        X_new = np.concatenate((X_normal[normal_indices],
                                X_anomaly[anomaly_indices]))

        y_new = np.concatenate((y_normal[normal_indices],
                                y_anomaly[anomaly_indices]))

        X = X_new
        y = y_new

        X = torch.tensor(X, device='cuda', dtype=torch.float32)
        entropys = ad.calculate_entropy_test(X, tau=tau).cpu().detach().numpy()

        entropys = np.nan_to_num(entropys, 0)

        # 按照标签绘制直方图
        plt.cla()
        plt.rcParams['font.sans-serif'] = ['SimSun']
        # plt.figure(figsize=(8, 6))
        plt.hist(x=entropys[y == 0], bins=50, alpha=0.5, label='正常', color='blue', range=(0, 2))
        plt.hist(x=entropys[y == 1], bins=50, alpha=0.5, label='故障', color='red', range=(0, 2))

        plt.title(f"正常样本和对抗样本的熵的频率分布")
        plt.xlabel("熵")
        plt.ylabel("频率")
        plt.legend()
        # plt.tight_layout()
        plt.show()
        plt.savefig(path_plot)


check_test_data_entropy()
check_entropy_aug()
