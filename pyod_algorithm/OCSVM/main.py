import gc
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.models.ocsvm import OCSVM


def load_data(dataInput):
    df = pd.read_csv(dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    return data, labels


def run(seed=0, input_path='', path_results='', nu=0.5):
    train_files = glob(os.path.join(input_path, '*train.csv'))
    test_files = glob(os.path.join(input_path, '*test.csv'))

    os.makedirs(os.path.join(path_results, f'{seed}/models'), exist_ok=True)
    os.makedirs(os.path.join(path_results, f'{seed}/scores'), exist_ok=True)
    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        data, labels = load_data(file)

        clf = OCSVM(nu=nu, random_state=seed)
        clf.fit(data)

    auc_roc_list = []
    auc_pr_list = []
    for id, file in enumerate(test_files):
        print(f"Test file {id}: {file}")
        data, labels = load_data(file)

        scores = clf.decision_function(data)  # Outlier scores for test data
        scores.tofile(os.path.join(path_results, f"{seed}/scores/anomaly_scores-{id}.csv"), sep="\n")

        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)

        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)

    # 创建包含AUC-ROC和AUC-PR值的DataFrame
    result = pd.DataFrame({'AUC-ROC': auc_roc_list, 'AUC-PR': auc_pr_list})

    # 计算均值
    mean_auc_roc = result['AUC-ROC'].mean()
    mean_auc_pr = result['AUC-PR'].mean()

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'AUC-ROC': [mean_auc_roc], 'AUC-PR': [mean_auc_pr]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    # 保存结果到CSV文件
    results_csv_path = os.path.join(path_results, f'{seed}/results.csv')
    result.to_csv(results_csv_path, index=True)  # index=True保留索引，这样'Mean'也会被写入文件

    print('Results saved.')

def run_1_by_1(seed=0, input_path='', path_results='', nu=0.5):
    train_files = glob(os.path.join(input_path, '*train.csv'))
    test_files = glob(os.path.join(input_path, '*test.csv'))

    os.makedirs(os.path.join(path_results, f'{seed}/models'), exist_ok=True)
    os.makedirs(os.path.join(path_results, f'{seed}/scores'), exist_ok=True)

    auc_roc_list = []
    auc_pr_list = []

    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        data, labels = load_data(file)

        clf = OCSVM(nu=nu, random_state=seed)
        clf.fit(data)

        file = test_files[id]
        print(f"Test file {id}: {file}")
        data, labels = load_data(file)

        scores = clf.decision_function(data)  # Outlier scores for test data
        scores.tofile(os.path.join(path_results, f"{seed}/scores/anomaly_scores-{id}.csv"), sep="\n")

        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)

        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)

        del clf
        gc.collect()

    # 创建包含AUC-ROC和AUC-PR值的DataFrame
    result = pd.DataFrame({'AUC-ROC': auc_roc_list, 'AUC-PR': auc_pr_list})

    # 计算均值
    mean_auc_roc = result['AUC-ROC'].mean()
    mean_auc_pr = result['AUC-PR'].mean()

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'AUC-ROC': [mean_auc_roc], 'AUC-PR': [mean_auc_pr]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    # 保存结果到CSV文件
    results_csv_path = os.path.join(path_results, f'{seed}/results.csv')
    result.to_csv(results_csv_path, index=True)  # index=True保留索引，这样'Mean'也会被写入文件

    print('Results saved.')

path_project = '/home/yukina/Missile_Fault_Detection/project_data'

if __name__ == '__main__':
    suffix = 'window=10, step=2'
    dataset_name = 'TLM-RATE'
    nu = 0.5
    path_data = os.path.join(path_project, f'data/{dataset_name}/csv/{suffix}')
    path_results = os.path.join(path_project,
                                f'pyod_algorithm/OCSVM/results/{dataset_name}/{suffix}/nu={nu}')

    all_scores = []
    AUCROC_seed = []
    AUCPR_seed = []
    for seed in range(3):
        if dataset_name == 'SMD':
            run_1_by_1(seed, path_data, path_results, nu=nu)
        else:
            run(seed, path_data, path_results, nu=nu)
        result = pd.read_csv(os.path.join(path_results, f'{seed}/results.csv'), index_col=0)
        AUCROC_seed.append(result['AUC-ROC'].values)
        AUCPR_seed.append(result['AUC-PR'].values)

        scores = pd.read_csv(os.path.join(path_results, f'{seed}/scores/anomaly_scores-0.csv'), header=None)
        scores = scores.to_numpy().mean(axis=1)
        all_scores.append(scores)
    all_scores = np.array(all_scores)
    mean_scores = all_scores.mean(axis=0)

    # 计算AUC-ROC和AUC-PR
    mean_aucroc = np.mean(AUCROC_seed, axis=0)[:-1] * 100
    std_aucroc = np.std(AUCROC_seed, axis=0)[:-1] * 100
    mean_aucpr = np.mean(AUCPR_seed, axis=0)[:-1] * 100
    std_aucpr = np.std(AUCPR_seed, axis=0)[:-1] * 100

    # 输出为csv文件
    np.save(os.path.join(path_results, 'scores.npy'), mean_scores)

    result = pd.DataFrame(
        {'mean_AUCROC': mean_aucroc, 'std_AUCROC': std_aucroc, 'mean_AUCPR': mean_aucpr, 'std_AUCPR': std_aucpr})

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'mean_AUCROC': [np.mean(mean_aucroc)], 'std_AUCROC': [np.mean(std_aucroc)],
                             'mean_AUCPR': [np.mean(mean_aucpr)], 'std_AUCPR': [np.mean(std_aucpr)]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    result.to_csv(os.path.join(path_results, 'results.csv'), index=True)
