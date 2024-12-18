import gc
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.models.auto_encoder import AutoEncoder
import argparse
from oc_gan import OCGAN

def load_data(dataInput):
    df = pd.read_csv(dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    return data, labels


def run(input_path='', path_results='', args=None):
    train_files = glob(os.path.join(input_path, '*train.csv'))
    test_files = glob(os.path.join(input_path, '*test.csv'))

    os.makedirs(os.path.join(path_results, f'{seed}/models'), exist_ok=True)
    os.makedirs(os.path.join(path_results, f'{seed}/scores'), exist_ok=True)
    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        train_data, train_labels = load_data(file)
        test_data, test_labels = load_data(test_files[0])
        clf = OCGAN(args)
        clf.fit(train_data, train_labels, test_data, test_labels)
        generated_data = clf.generate_samples(num_samples=1000)
        np.savez(os.path.join(path_results, f'{seed}/generated_data_{id}.npz'), X=generated_data)

    auc_roc_list = []
    auc_pr_list = []
    for id, file in enumerate(test_files):
        print(f"Test file {id}: {file}")
        data, labels = load_data(file)

        scores = clf.decision_function(data)  # Outlier scores for test data
        scores.tofile(os.path.join(path_results, f"{seed}/scores/anomaly_scores-{id}.csv"), sep="\n")

        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores, pos_label=1)

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

def run_1_by_1(input_path='', path_results='', args=None):
    train_files = glob(os.path.join(input_path, '*train.csv'))
    test_files = glob(os.path.join(input_path, '*test.csv'))

    os.makedirs(os.path.join(path_results, f'{seed}/models'), exist_ok=True)
    os.makedirs(os.path.join(path_results, f'{seed}/scores'), exist_ok=True)

    auc_roc_list = []
    auc_pr_list = []

    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        train_data, train_labels = load_data(file)
        test_data, test_labels = load_data(test_files[0])

        clf = OCGAN(args)
        clf.fit(train_data, train_labels, test_data, test_labels)

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

def get_dataset_config(dataset_name):
    if dataset_name == 'Metro':
        img_shape = 5
        latent_dim = 5
        suffix = 'window=1, step=1'

    elif dataset_name == 'SMD_group4':
        img_shape = 180
        latent_dim = 50
        suffix = 'window=100, step=10'
    elif dataset_name == 'SWAT':
        img_shape = 255
        latent_dim = 200
        suffix = 'window=20, step=1'
    elif dataset_name == 'GHL':
        img_shape = 80
        latent_dim = 80
        suffix = 'window=100, step=10'
    elif dataset_name == 'TLM-RATE':
        img_shape = 48
        latent_dim = 48
        suffix = 'window=10, step=2'
    else:
        return None, None
    return img_shape, latent_dim, suffix

if __name__ == '__main__':
    dataset_name = 'GHL'
    img_shape, latent_dim, suffix = get_dataset_config(dataset_name)
    epochs = 40

    path_data = os.path.join(path_project, f'data/{dataset_name}/csv/{suffix}')
    prefix = f'epochs={epochs}, latent_dim={latent_dim}'
    path_results = os.path.join(path_project,
                                f'OCAN-master/results/{dataset_name}/{suffix}', prefix)


    all_scores = []
    AUCROC_seed = []
    AUCPR_seed = []
    for seed in range(3):
        parser = argparse.ArgumentParser('Train your OCGAN')

        ###Training hyperparameter
        parser.add_argument('--dataset', type=str, default=dataset_name, help='mnist | cifar10')
        parser.add_argument('--en_ae', type=int, default=1, help='network config id')
        parser.add_argument('--epochs', type=int, default=epochs, help='number of epochs to train')
        parser.add_argument('--latent_dim', type=int, default=latent_dim,
                                   help='Latent dimension of Gaussian noise input to Generator')
        parser.add_argument('--img_shape', type=int, default=img_shape)

        args = parser.parse_args()


        if dataset_name == 'SMD':
            run_1_by_1(path_data, path_results, args)
        else:
            run(path_data, path_results, args)
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

    print(prefix)