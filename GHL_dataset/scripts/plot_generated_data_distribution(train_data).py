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

iteration = 1
test_set_name = 'GHL'
model_name = 'WGAN-GP, euc, window=100, step=10, no_tau2_K=7,deepsad_epoch=1,gan_epoch=1,lam1=0,lam2=0,lam3=0,latent_dim=80,lr=0.0002,clip_value=0.01,lambda_gp=1,seed=0'
output_dir = os.path.join(path_project, f'GHL_dataset/log/{test_set_name}/train_result', model_name)

train_dir = os.path.join(path_project, f'data/{test_set_name}/yukina_data/ensemble_data, window=100, step=10', 'init', 'K=7')

detector_dir = os.path.join(path_project, f'GHL_dataset/models/{test_set_name}/ensemble', model_name)

test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data, window=100, step=10')

dir_model = os.path.join(path_project, f'GHL_dataset/models/{test_set_name}/ensemble', model_name)
for dataset_name in os.listdir(test_data_path)[:1]:
    for iteration in range(0, 49):
        print(f'Plotting {dataset_name} with iteration {iteration}...')
        path_train = os.path.join(train_dir)
        path_plot = os.path.join(output_dir, 'generated_data', dataset_name)
        os.makedirs(path_plot, exist_ok=True)
        path_detector = os.path.join(detector_dir, f'{iteration}')

        # plot时是否考虑聚类标签
        use_train_cluster_label = False
        # 随机抽取的样本数
        num_samples = 2000
        np.random.seed(0)

        datasets = []
        for dataset in os.listdir(path_train):
            datasets.append(np.load(os.path.join(path_train, dataset)))

        # 加载生成模型
        GAN_config = {"seed": 0,
        "latent_dim": 80,
        "lr": 0.0002,
        "clip_value": 0.01,
        "lambda_gp": 1000000,
        "n_epochs": 1,
        "lam1": 0,
        "lam2": 0,
        "lam3": 0,
        "tau1": 1,
        "img_size": 80
                      }
        generator = Adversarial_Generator(config=GAN_config)
        generator.load_model(os.path.join(dir_model, f'{iteration}'))
        gen_samples = generator.sample_generate(num=num_samples)
        generated_data = np.array(gen_samples.detach().cpu()).squeeze(1)



        data = np.load(os.path.join(test_data_path, dataset_name))
        dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
                    'X_test': data['X_test'], 'y_test': data['y_test']}
        anomaly_data = dataset['X_test'][np.where(dataset['y_test'] == 1)]
        sampled_anomaly = anomaly_data[np.random.choice(range(0, anomaly_data.shape[0]), num_samples, replace=True)]
        # faults = []
        # fault_data = []
        # normal_data = []
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
        #     sampled_anomaly = anomaly_data[np.random.choice(range(0, anomaly_data.shape[0]), num_samples, replace=True)]
        #     fault_data.append(sampled_anomaly)

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
            assert NotImplementedError
            # 正常数据的标签大于0，异常数据的标签小于0，生成数据的标签为0
            X_plot = np.concatenate((np.concatenate(sampled_train_data), sampled_anomaly))
            y_plot = np.concatenate((np.concatenate([np.ones(num_samples) * (id + 1) for id, fault in enumerate(sampled_train_data)]), -np.ones(num_samples)))
        else:
            # 不区分训练数据的聚类标签
            X_plot = np.concatenate((sampled_init_train_data, sampled_anomaly))
            y_plot = np.concatenate((np.ones(num_samples), -np.ones(num_samples)))

        sampled_generated = generated_data[np.random.choice(range(0, generated_data.shape[0]), num_samples, replace=True)]

        X_train = X_plot
        y_train = y_plot

        X_all = np.concatenate((X_plot, sampled_generated))
        y_all = np.concatenate((y_plot, np.zeros(sampled_generated.shape[0])))

        # import umap
        # umap_model = umap.UMAP(n_components=2, random_state=42)
        # X_2d = umap_model.fit_transform(X_train)

        tsne1 = TSNE(n_components=2, random_state=0)  # n_components表示目标维度

        # 创建MinMaxScaler对象
        scaler1 = MinMaxScaler()
        # 对数据进行归一化
        # normalized_data = scaler1.fit_transform(X_all)
        normalized_data = X_all
        X_2d = tsne1.fit_transform(normalized_data)  # 对数据进行降维处理

        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.figure(figsize=(12, 9))

        if use_train_cluster_label:
            for id, train_data in enumerate(sampled_train_data):
                plt.scatter(X_2d[y_all == (id + 1), 0], X_2d[y_all == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
        else:
            plt.scatter(X_2d[y_all == 1, 0], X_2d[y_all == 1, 1], label=f'正常数据', alpha=0.5)

        plt.scatter(X_2d[y_all == 0, 0], X_2d[y_all == 0, 1], label='生成数据', alpha=0.5)

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
        tsne2 = TSNE(n_components=2, random_state=0)  # n_components表示目标维度
        # 创建MinMaxScaler对象
        scaler1 = MinMaxScaler()
        # 对数据进行归一化
        # normalized_data = scaler1.fit_transform(X_all)
        normalized_data = X_all
        X_2d = tsne2.fit_transform(normalized_data)  # 对数据进行降维处理

        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.figure(figsize=(12, 9))

        if use_train_cluster_label:
            for id, train_data in enumerate(sampled_train_data):
                plt.scatter(X_2d[y_all == (id + 1), 0], X_2d[y_all == (id + 1), 1], label=f'正常数据_{id}', alpha=0.5)
        else:
            plt.scatter(X_2d[y_all == 1, 0], X_2d[y_all == 1, 1], label=f'正常数据', alpha=0.5)

        plt.scatter(X_2d[y_all == 0, 0], X_2d[y_all == 0, 1], label='生成数据', alpha=0.5)

        plt.scatter(X_2d[y_all == -1, 0], X_2d[y_all == -1, 1], label='故障数据', alpha=0.5)
        # for id, fault in enumerate(faults):
        #     plt.scatter(X_2d[y_all == -(id + 1), 0], X_2d[y_all == -(id + 1), 1], label=fault, alpha=0.5)

        # plt.xlim([-25, 25])
        # plt.ylim([-25, 25])
        plt.legend()

        plt.title('T-SNE 对真实数据和生成数据的可视化')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
        plt.savefig(os.path.join(path_plot, f'TSNE2 of Generated Data_train_{iteration}.png'))
        plt.close()