import argparse
import gc
import pickle

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm

from Daphnet_dataset.scripts.group_result_Daphnet import group_results
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
from glob import glob

from adversarial_ensemble_AD.data_generate.gan import Adversarial_Generator

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

def metric(y_true, y_score, pos_label=1):
    aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
    aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=1)

    return {'aucroc': aucroc, 'aucpr': aucpr, 'scores': y_score, 'labels': y_true}

def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]

    Parameters
    ----------
        score: np.ndarray
            anomaly score, higher score indicates higher likelihoods to be anomaly
        label: np.ndarray
            ground-truth label

    Return
    ----------
        score: np.ndarray
            adjusted score

    """

    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    train_set_name = 'WQ'
    parser.add_argument("--seed", type=int, default=3, help="seed")
    # parser.add_argument("--path_train_data", type=str,
    #                     default=os.path.join(path_project,
    #                                          f'data/{train_set_name}/yukina_data/DeepSAD_data, window=100, step=10'))
    parser.add_argument("--path_train_data", type=str,
                        default=os.path.join(path_project,
                                             f'data/{train_set_name}/yukina_data/DeepSAD_data, window=10, step=1'))
    parser.add_argument("--path_test", type=str,
                        default=os.path.join(path_project,
                                             f'data/{train_set_name}/yukina_data/DeepSAD_data, window=10, step=1'))
    parser.add_argument("--dir_model", type=str,
                        default=os.path.join(path_project, f'{train_set_name}_dataset/models/{train_set_name}/DeepSAD'))
    parser.add_argument("--path_output", type=str,
                        default=os.path.join(path_project, f'{train_set_name}_dataset/log/{train_set_name}/train_result'))
    parser.add_argument("--DeepSAD_config", type=dict, default={
        "n_epochs": 10,
        "ae_n_epochs": 10,
        "lr": 0.001,
        "ae_lr": 0.001,
        "net_name": 'Dense'
    }, help="config of DeepSAD")
    config = parser.parse_args()
    param_dir = f'fix_pretrain, net=Dense, std, window=10, step=1, n_epochs={config.DeepSAD_config["n_epochs"]}, ae_n_epochs={config.DeepSAD_config["ae_n_epochs"]}, lr={config.DeepSAD_config["lr"]}, ae_lr={config.DeepSAD_config["ae_lr"]}'

    config.dir_model = os.path.join(config.dir_model, param_dir)
    config.path_output = os.path.join(config.path_output, param_dir)

    data = np.load(os.path.join(config.path_train_data, 'train.npz'))
    if config.DeepSAD_config['net_name'] == 'Dense' or config.DeepSAD_config['net_name'] == 'Simple_Dense':
        X_train = data['X_train'].reshape(data['X_train'].shape[0], -1)
    else:
        X_train = data['X_train']
    y_train = data['y_train']

    test_files = glob(os.path.join(config.path_test, '*test.npz'))

    for test_path in tqdm(test_files, desc='Total progress'):
        base_name = os.path.basename(test_path).replace('.test.npz', '')
        # 创建结果文件夹路径

        data = np.load(os.path.join(config.path_train_data, test_path))

        ys = []
        scores_seed = []
        for seed in tqdm(range(0, config.seed), desc='Seed progress'):
            config.DeepSAD_config["loss_output_path"] = os.path.join(config.path_output, 'deepsad_loss', f'seed={seed}')
            os.makedirs(config.DeepSAD_config["loss_output_path"], exist_ok=True)

            path_save = os.path.join(path_project, f'{train_set_name}_dataset/log/{train_set_name}/DeepSAD', param_dir,
                                     base_name, f'seed={seed}')
            os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

            if not config.dir_model is None:
                os.makedirs(config.dir_model, exist_ok=True)
                if not os.path.exists(os.path.join(config.dir_model,
                                                   f'DeepSAD_seed={seed}.pth')):

                    model = DeepSAD(seed=seed, config=config.DeepSAD_config)
                    model.fit(X_train=X_train, y_train=y_train)
                    model.deepSAD.save_model(export_model=os.path.join(config.dir_model,
                                                                          f'DeepSAD_seed={seed}.pth'),
                                                save_ae=True)
                else:
                    model = DeepSAD(seed=seed, load_model=os.path.join(config.dir_model,
                                                                      f'DeepSAD_seed={seed}.pth'), config=config.DeepSAD_config)
                    model.load_model_from_file(input_size=X_train.shape[1])
            else:
                model = DeepSAD(seed=seed, config=config.DeepSAD_config)
                model.fit(X_train=X_train, y_train=y_train)


            # 评估集成模型
            if config.DeepSAD_config['net_name'] == 'Dense' or config.DeepSAD_config['net_name'] == 'Simple_Dense':
                X_test = data['X_test'].reshape(data['X_test'].shape[0], -1)
            else:
                X_test = data['X_test']

            y_test = data['y_test']

            start_time = time.time()

            scores, outputs = model.predict_score(X_test)
            scores = np.array(scores)

            end_time = time.time()
            time_inference = end_time - start_time

            precision, recall, _ = precision_recall_curve(y_test, scores)
            precision_threshold = 0.95
            recall_at_threshold = recall[np.where(precision >= precision_threshold)[0][0]]
            recall_threshold = 0.95
            precision_at_threshold = precision[np.where(recall >= recall_threshold)[0][-1]]

            # performance
            result_1 = metric(y_true=y_test, y_score=scores, pos_label=1)
            result_2 = metric(y_true=y_test, y_score=adjust_scores(y_test, scores),
                              pos_label=1)
            result = {'aucroc': result_1['aucroc'],
                      'aucpr': result_1['aucpr'],
                      'FDR_at_threshold': recall_at_threshold,
                      'FAR_at_threshold': 1 - precision_at_threshold,
                      'time_inference': time_inference,
                      'adjust_aucroc': result_2['aucroc'],
                      'adjust_aucpr': result_2['aucpr'],
                      }

            # 创建DataFrame对象
            df = pd.DataFrame(list(result.items()), columns=['指标', '分数'])
            # 将DataFrame数据输出到CSV文件
            df.to_csv(os.path.join(path_save, f'result.csv'), index=False)
            scores_seed.append(result_1['scores'])
            ys.append(result_1['labels'])

        path_score_output = os.path.join(config.path_output, f'scores/{base_name}')
        os.makedirs(path_score_output, exist_ok=True)
        # 异常分数保存到npy文件
        np.save(os.path.join(path_score_output, f"scores_DeepSAD.npy"), np.array(scores_seed).mean(axis=0))
        np.save(os.path.join(path_score_output, f"labels_DeepSAD.npy"), np.array(ys).mean(axis=0))

    group_results(os.path.join(path_project, f'{train_set_name}_dataset/log/{train_set_name}/DeepSAD', param_dir))