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
from scripts.group_result_ensemble import group_results
# from scripts.combine_seed_result import combine_seed_results
# from scripts.combine_epoch_result import combine_epoch_results

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'


def metric(y_true, y_score, pos_label=1):
    aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
    aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=1)

    return {'aucroc': aucroc, 'aucpr': aucpr, 'scores': y_score, 'labels': y_true}


def ensemble_test(model_name):
    seed = 0
    n_samples_threshold = 0

    iterations = range(0, 30)

    for iteration in iterations:
        print(f"Start iteration {iteration}")
        test_set_name = 'GHL'
        model_path = os.path.join(path_project, f'{test_set_name}_dataset/models/{test_set_name}/ensemble/{model_name}/{iteration}')
        train_data_path = os.path.join(path_project,
                                       f'data/{test_set_name}/yukina_data/ensemble_data, window=100, step=10/init/K=7')
        test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data, window=100, step=10, type2')
        output_path = os.path.join(path_project,
                                   f'{test_set_name}_dataset/log/{test_set_name}/ensemble/DeepSAD/type2/{model_name}/{iteration}')
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

        train_data_example = np.load(os.path.join(train_data_path, os.listdir(train_data_path)[0]))

        # 加载模型
        model_list = []
        for model_file in os.listdir(model_path):
            if not model_file.startswith('DeepSAD'):
                continue
            model = DeepSAD(seed=seed, load_model=os.path.join(model_path, model_file))
            model.load_model_from_file(input_size=train_data_example['X_train'].shape[1])
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

        thresholds = np.percentile(score_train, 95)

        score_ensemble_list = []
        y_list = []
        # 遍历所有数据集文件
        for test_set in tqdm(os.listdir(test_data_path)[:-1], desc='Total progress'):
            base_name = os.path.basename(test_set).replace('.npz', '')
            # 创建结果文件夹路径

            # 加载数据集
            FDRs = []
            FARs = []
            scores_ensemble = []
            ys = []

            path_test_set = os.path.join(test_data_path, test_set)

            data = np.load(path_test_set)
            dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
                       'X_test': data['X_test'], 'y_test': data['y_test']}

            path_save = os.path.join(output_path, base_name)
            os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

            # 评估集成模型
            start_time = time.time()
            score_list = []
            for model in model_list:
                score_seperate, outputs = model.predict_score(dataset['X_test'])
                score_list.append(score_seperate)
            score_ensemble = np.array(score_list).mean(axis=0)

            id_anomaly_pred = np.where(score_ensemble > thresholds)[0]
            id_normal_pred = np.where(score_ensemble <= thresholds)[0]

            end_time = time.time()
            time_inference = end_time - start_time

            tp = np.size(np.where(dataset['y_test'][id_anomaly_pred] == 1)[0], 0)
            fp = np.size(np.where(dataset['y_test'][id_anomaly_pred] == 0)[0], 0)
            fn = np.size(np.where(dataset['y_test'][id_normal_pred] == 1)[0], 0)

            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            precision_list, recall_list, _ = precision_recall_curve(dataset['y_test'], score_ensemble)
            precision_threshold = 0.95
            recall_at_threshold = recall_list[np.where(precision_list >= precision_threshold)[0][0]]
            recall_threshold = 0.95
            precision_at_threshold = precision_list[np.where(recall_list >= recall_threshold)[0][-1]]

            # performance
            result_1 = metric(y_true=dataset['y_test'], y_score=score_ensemble, pos_label=1)
            result = {'aucroc': result_1['aucroc'],
                      'aucpr': result_1['aucpr'],
                      'precision': precision,
                      'recall': recall,
                      'FDR_at_threshold': recall_at_threshold,
                      'FAR_at_threshold': 1 - precision_at_threshold,
                      'time_inference': time_inference
                      }

            # 创建DataFrame对象
            df = pd.DataFrame(list(result.items()), columns=['指标', '分数'])
            # 将DataFrame数据输出到CSV文件
            df.to_csv(os.path.join(path_save, f'result.csv'), index=False)
            scores_ensemble.append(result_1['scores'])
            ys.append(result_1['labels'])

        score_ensemble_list.append(scores_ensemble)
        y_list.append(ys)

        path_score_output = os.path.join(path_project, f'{test_set_name}_dataset/log/{test_set_name}/train_result',
                                         f'{model_name}/scores/{iteration}')
        os.makedirs(path_score_output, exist_ok=True)
        # 异常分数保存到npy文件
        np.save(os.path.join(path_score_output, f"scores_DeepSAD.npy"), np.array(scores_ensemble).mean(axis=0))
        np.save(os.path.join(path_score_output, f"labels_DeepSAD.npy"), np.array(ys).mean(axis=0))

        group_results(output_path)

    # base_dir = os.path.join(path_project, f'{test_set_name}_dataset/log/{test_set_name}/ensemble/DeepSAD')
    # prefix = model_name.rsplit(',', 1)[0]
    #
    # combine_seed_results(base_dir, prefix)
    #
    # seed_dir = os.path.join(base_dir, 'seed_group', prefix)
    # combine_epoch_results(seed_dir)


    print("All down!")

if __name__ == '__main__':
    model_name = f'WGAN-GP, euc, window=100, step=10, no_tau2_K=7,deepsad_epoch=1,gan_epoch=1,lam1=2000,lam2=300,lam3=0,latent_dim=80,lr=0.0002,clip_value=0.01,lambda_gp=1000000,seed=2'
    ensemble_test(model_name)