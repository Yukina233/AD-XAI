import argparse

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm

from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import glob
from adbench_modified.myutils import Utils
from adversarial_ensemble_AD.data_generate.gan import Adversarial_Generator
from adbench_modified.run import RunPipeline
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD
from adversarial_ensemble_AD.scripts.group_result_ensemble import group_results

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'


def metric(y_true, y_score, pos_label=1):
    aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
    aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=1)

    return {'aucroc': aucroc, 'aucpr': aucpr, 'scores': y_score, 'labels': y_true}


if __name__ == '__main__':
    seed = 3
    n_samples_threshold = 0

    iteration = 0
    test_set_name = 'banwuli_data'
    model_name = 'no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=100,tau1=0.1'
    model_path = os.path.join(path_project, f'adversarial_ensemble_AD/models/{test_set_name}/ensemble/{model_name}/{iteration}')
    train_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/train_seperate/init/K=7')
    test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data')
    output_path = os.path.join(path_project, f'adversarial_ensemble_AD/log/{test_set_name}/train_result/{model_name}/log_seperate_model/{iteration}')
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

    # 加载模型
    model_list = []
    for model_file in os.listdir(model_path):
        model = DeepSAD(seed=seed, load_model=os.path.join(model_path, model_file))
        model.load_model_from_file()
        model_list.append(model)

    # 计算阈值
    X_train = None
    y_train = None
    for train_dataset in os.listdir(train_data_path):
        data = np.load(os.path.join(train_data_path, train_dataset))
        if X_train is None:
            X_train = data['X_train']
            y_train = data['y_train']
        else:
            X_train = np.concatenate((X_train, data['X_train']))
            y_train = np.concatenate((y_train, data['y_train']))

    score_list = []
    for model in model_list:
        score_seperate, outputs = model.predict_score(X_train)
        score_list.append(score_seperate)
    score_train = np.array(score_list).mean(axis=0)

    thresholds = np.percentile(score_train, 100 * (1 - sum(y_train) / len(y_train)))
    # thresholds = thresholds * 1.6

    # 测试每个子模型
    for model_id, model in enumerate(model_list):

        score_ensemble_list = []
        y_list = []
        # 遍历所有数据集文件
        for fault in tqdm(os.listdir(test_data_path), desc='Total progress'):
            base_name = os.path.basename(fault).replace('.npz', '')
            # 创建结果文件夹路径

            # 加载数据集
            FDRs = []
            FARs = []

            path_fault = os.path.join(test_data_path, fault)
            files = os.listdir(path_fault)

            for i in range(1, int(len(files) + 1)):

                data = np.load(os.path.join(path_fault, f"dataset_{i}.npz"))
                dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
                           'X_test': data['X_test'], 'y_test': data['y_test']}

                path_save = os.path.join(output_path, f'model_{model_id}', base_name)
                os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

                # 评估集成模型
                start_time = time.time()

                score_seperate, outputs = model.predict_score(dataset['X_test'])

                id_anomaly_pred = np.where(score_seperate > thresholds)[0]
                id_normal_pred = np.where(score_seperate <= thresholds)[0]

                end_time = time.time()
                time_inference = end_time - start_time

                tp = np.size(np.where(dataset['y_test'][id_anomaly_pred] == 1)[0], 0)
                fp = np.size(np.where(dataset['y_test'][id_anomaly_pred] == 0)[0], 0)
                fn = np.size(np.where(dataset['y_test'][id_normal_pred] == 1)[0], 0)

                FDR = tp / (tp + fn)
                if tp + fp == 0:
                    FAR = 0
                else:
                    FAR = fp / (tp + fp)

                precision, recall, _ = precision_recall_curve(dataset['y_test'], score_seperate)
                precision_threshold = 0.999
                recall_at_threshold = recall[np.where(precision >= precision_threshold)[0][0]]
                recall_threshold = 0.999
                precision_at_threshold = precision[np.where(recall >= recall_threshold)[0][-1]]

                # performance
                result_1 = metric(y_true=dataset['y_test'], y_score=score_seperate, pos_label=1)
                result = {'aucroc': result_1['aucroc'],
                          'aucpr': result_1['aucpr'],
                          'FDR': FDR,
                          'FAR': FAR,
                          'FDR_at_threshold': recall_at_threshold,
                          'FAR_at_threshold': 1 - precision_at_threshold,
                          'time_inference': time_inference
                          }


                # 创建DataFrame对象
                df = pd.DataFrame(list(result.items()), columns=['指标', '分数'])
                # 将DataFrame数据输出到CSV文件
                df.to_csv(os.path.join(path_save, f'{i}.csv'), index=False)

        group_results(os.path.join(output_path, f'model_{model_id}'))

    print("All down!")
